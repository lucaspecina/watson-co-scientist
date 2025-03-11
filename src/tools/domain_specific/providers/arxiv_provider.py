"""
ArXiv knowledge provider for physics, computer science, mathematics, and related fields.
"""

import os
import json
import re
import logging
import httpx
import asyncio
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union

from ..base_provider import DomainKnowledgeProvider

logger = logging.getLogger("co_scientist")

# COMPLETE BYPASS OF FEEDPARSER - NO SGMLLIB DEPENDENCY
# This implements a minimal version of feedparser functionality
class AttrDict(dict):
    """Dictionary that allows attribute access to its keys."""
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def text_content(element):
    """Extract text content from an XML element recursively."""
    return (element.text or '') + ''.join(text_content(e) + (e.tail or '') for e in element)

def parse_feed(content):
    """Minimal feedparser replacement that works without sgmllib."""
    try:
        # Try to parse as XML
        if isinstance(content, str):
            try:
                root = ET.fromstring(content)
            except ET.ParseError:
                return create_empty_feed()
        else:
            # If content is file-like, read the contents
            try:
                content_text = content.read()
                if isinstance(content_text, bytes):
                    content_text = content_text.decode('utf-8')
                root = ET.fromstring(content_text)
            except:
                return create_empty_feed()
        
        # Create feed dictionary
        feed = AttrDict()
        feed.entries = []
        feed.feed = AttrDict()
        feed.feed.title = ""
        feed.feed.link = ""
        feed.bozo = False
        
        # Try to determine if it's Atom or RSS
        namespace = root.tag.split('}')[0] + '}' if '}' in root.tag else ''
        
        if root.tag == 'rss' or root.tag.endswith('}rss'):
            # RSS feed
            channel = root.find('./channel') or root
            
            # Feed info
            title_elem = channel.find('./title') or channel.find(f'{namespace}title')
            if title_elem is not None:
                feed.feed.title = text_content(title_elem)
            
            link_elem = channel.find('./link') or channel.find(f'{namespace}link')
            if link_elem is not None:
                feed.feed.link = text_content(link_elem)
            
            # Entries
            for item in channel.findall('./item') or channel.findall(f'{namespace}item'):
                entry = AttrDict()
                
                # Title
                title_elem = item.find('./title') or item.find(f'{namespace}title')
                if title_elem is not None:
                    entry.title = text_content(title_elem)
                else:
                    entry.title = ""
                
                # Link
                link_elem = item.find('./link') or item.find(f'{namespace}link')
                if link_elem is not None:
                    entry.link = text_content(link_elem)
                else:
                    entry.link = ""
                
                # Description/summary
                desc_elem = item.find('./description') or item.find(f'{namespace}description')
                if desc_elem is not None:
                    entry.summary = text_content(desc_elem)
                else:
                    entry.summary = ""
                
                # ID
                id_elem = item.find('./guid') or item.find(f'{namespace}guid')
                if id_elem is not None:
                    entry.id = text_content(id_elem)
                else:
                    entry.id = entry.link
                
                feed.entries.append(entry)
                
        elif root.tag == 'feed' or root.tag.endswith('}feed'):
            # Atom feed
            # Feed info
            title_elem = root.find('./title') or root.find(f'{namespace}title')
            if title_elem is not None:
                feed.feed.title = text_content(title_elem)
            
            link_elem = root.find("./link[@rel='alternate']") or root.find(f"{namespace}link[@rel='alternate']") or root.find('./link') or root.find(f'{namespace}link')
            if link_elem is not None:
                feed.feed.link = link_elem.get('href', '')
            
            # Entries
            for entry_elem in root.findall('./entry') or root.findall(f'{namespace}entry'):
                entry = AttrDict()
                
                # Title
                title_elem = entry_elem.find('./title') or entry_elem.find(f'{namespace}title')
                if title_elem is not None:
                    entry.title = text_content(title_elem)
                else:
                    entry.title = ""
                
                # Link
                link_elem = entry_elem.find("./link[@rel='alternate']") or entry_elem.find(f"{namespace}link[@rel='alternate']") or entry_elem.find('./link') or entry_elem.find(f'{namespace}link')
                if link_elem is not None:
                    entry.link = link_elem.get('href', '')
                else:
                    entry.link = ""
                
                # Summary
                content_elem = entry_elem.find('./content') or entry_elem.find(f'{namespace}content')
                summary_elem = entry_elem.find('./summary') or entry_elem.find(f'{namespace}summary')
                
                if content_elem is not None:
                    entry.summary = text_content(content_elem)
                elif summary_elem is not None:
                    entry.summary = text_content(summary_elem)
                else:
                    entry.summary = ""
                
                # ID
                id_elem = entry_elem.find('./id') or entry_elem.find(f'{namespace}id')
                if id_elem is not None:
                    entry.id = text_content(id_elem)
                else:
                    entry.id = entry.link
                
                feed.entries.append(entry)
        
        return feed
        
    except Exception as e:
        # Create empty feed on error
        feed = create_empty_feed()
        feed.bozo = True
        feed.bozo_exception = str(e)
        return feed

def create_empty_feed():
    """Create an empty feed structure."""
    feed = AttrDict()
    feed.entries = []
    feed.feed = AttrDict()
    feed.feed.title = ""
    feed.feed.link = ""
    feed.bozo = True
    return feed

class ArxivProvider(DomainKnowledgeProvider):
    """
    ArXiv knowledge provider for physics, computer science, mathematics, and related fields.
    
    This provider interfaces with the ArXiv API to search and retrieve
    preprints from various scientific domains including physics, computer science, 
    mathematics, quantitative biology, quantitative finance, and statistics.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the ArXiv provider.
        
        Args:
            config (Dict[str, Any], optional): Configuration for the provider. Defaults to None.
        """
        # ArXiv covers multiple domains
        super().__init__(domain="multi_domain", config=config)
        
        # ArXiv API base URL
        self.base_url = "http://export.arxiv.org/api"
        
        # HTTP client for API requests
        self.client = httpx.AsyncClient(timeout=30.0)
        
        # ArXiv categories by domain
        self.domain_categories = {
            "physics": ["physics", "astro-ph", "cond-mat", "gr-qc", "hep-ex", "hep-lat", "hep-ph", "hep-th", "math-ph", "nucl-ex", "nucl-th", "quant-ph"],
            "computer_science": ["cs"],
            "mathematics": ["math"],
            "biology": ["q-bio"],
            "finance": ["q-fin"],
            "statistics": ["stat"]
        }
        
        # Citation format templates
        self.citation_templates = {
            "apa": "{authors} ({year}). {title}. {journal}, {identifier}. {url}",
            "mla": "{authors}. \"{title}.\" {journal}, {year}, {identifier}. {url}",
            "chicago": "{authors}. \"{title}.\" {journal} ({year}): {identifier}. {url}",
            "vancouver": "{authors}. {title}. {journal}. {year};{identifier}. {url}"
        }
    
    async def initialize(self) -> bool:
        """
        Initialize the ArXiv provider.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        try:
            # Test connection to ArXiv API with a simple query
            # Using a known field and a common term that should always return results
            url = f"{self.base_url}/query"
            params = {
                "search_query": "ti:quantum", # Search in title for "quantum"
                "start": "0",
                "max_results": "1"
            }
                
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            
            # Parse response with feedparser
            response_text = response.text
            feed = parse_feed(response_text)
            
            # More robust check: check if feed has entries or valid structure
            if hasattr(feed, 'entries') and len(feed.entries) > 0:
                self._is_initialized = True
                logger.info("ArXiv provider initialized successfully")
                return True
            elif hasattr(feed, 'feed') and 'opensearch_totalresults' in feed.feed:
                # Check if it's a valid response but with no results
                self._is_initialized = True
                logger.info("ArXiv provider initialized successfully (no results for test query)")
                return True
            elif hasattr(feed, 'bozo') and feed.bozo == 0:
                # Feed parsed without errors, assume it's valid even if structured differently
                self._is_initialized = True
                logger.info("ArXiv provider initialized successfully (valid feed but unexpected structure)")
                return True
            else:
                # Log detailed debugging information
                logger.warning("ArXiv API response did not contain expected structure, but we'll attempt to continue")
                # Force initialization despite the unexpected structure
                self._is_initialized = True
                
                # Log debug info for diagnosis
                logger.debug(f"Feed keys: {feed.keys() if hasattr(feed, 'keys') else 'No keys attribute'}")
                logger.debug(f"Feed has entries: {hasattr(feed, 'entries')}")
                if hasattr(feed, 'entries'):
                    logger.debug(f"Entries length: {len(feed.entries)}")
                logger.debug(f"Response status code: {response.status_code}")
                logger.debug(f"Response headers: {response.headers}")
                # Only log a snippet of response text to avoid overloading logs
                logger.debug(f"Response text snippet: {response_text[:200]}...")
                return True
                
        except Exception as e:
            logger.error(f"Failed to initialize ArXiv provider: {str(e)}")
            # Log traceback for debugging
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return False
    
    async def search(self, query: str, limit: int = 10, domain: str = None) -> List[Dict[str, Any]]:
        """
        Search ArXiv for scientific papers.
        
        Args:
            query (str): The search query
            limit (int, optional): Maximum number of results. Defaults to 10.
            domain (str, optional): Specific domain to search (physics, computer_science, etc). 
                                   Defaults to None (search all categories).
            
        Returns:
            List[Dict[str, Any]]: List of search results
        """
        if not self._is_initialized:
            initialized = await self.initialize()
            if not initialized:
                logger.warning("Failed to initialize ArXiv provider, returning empty search results")
                return []
            
        try:
            # Format the search query to be compatible with ArXiv's syntax
            # ArXiv expects queries in the format: field:term
            # Common fields: all, ti (title), au (author), abs (abstract), cat (category)
            
            # Clean the query - replace special characters and prepare for ArXiv API
            clean_query = query.replace("'", " ").replace('"', " ").strip()
            
            # Handle completely empty queries
            if not clean_query:
                logger.warning("Empty search query provided to ArXiv provider")
                return []
                
            # If the query is a simple search term without field specifications,
            # search in title and abstract which is more specific
            if ":" not in clean_query:
                search_query = f"(ti:{clean_query} OR abs:{clean_query})"
            else:
                # User provided a field-specific query, use as-is
                search_query = clean_query
            
            # If domain specified, limit to relevant categories
            if domain and domain in self.domain_categories:
                categories = self.domain_categories[domain]
                category_terms = []
                for cat in categories:
                    # Handle wildcards properly
                    if cat.endswith("*"):
                        category_terms.append(f"cat:{cat}")
                    else:
                        category_terms.append(f"cat:{cat}")
                
                if category_terms:
                    category_query = " OR ".join(category_terms)
                    search_query = f"({search_query}) AND ({category_query})"
            
            logger.info(f"Searching ArXiv with query: {search_query}")
            
            # Call ArXiv API
            url = f"{self.base_url}/query"
            params = {
                "search_query": search_query,
                "start": "0",
                "max_results": str(limit),
                "sortBy": "relevance",
                "sortOrder": "descending"
            }
                
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            
            # Parse response (ArXiv returns Atom feed XML)
            response_text = response.text
            feed = parse_feed(response_text)
            
            # Check if feed has entries
            if not hasattr(feed, 'entries'):
                logger.warning(f"ArXiv API response doesn't contain entries. Response headers: {response.headers}")
                logger.debug(f"Response text snippet: {response_text[:200]}...")
                return []
            
            # Process results
            results = []
            for entry in feed.entries:
                try:
                    # Extract authors
                    authors = []
                    if hasattr(entry, 'authors'):
                        for author in entry.authors:
                            if hasattr(author, 'name'):
                                authors.append(author.name)
                    elif 'author' in entry:  # Sometimes there's a single author field
                        authors = [entry.author]
                    
                    # Extract categories
                    categories = []
                    if hasattr(entry, 'tags'):
                        for category in entry.tags:
                            if hasattr(category, 'term'):
                                categories.append(category.term)
                    
                    # Extract ID - format it correctly
                    arxiv_id = ""
                    if hasattr(entry, 'id'):
                        # Try to extract ID from the full identifier
                        id_parts = entry.id.split('/')
                        if len(id_parts) > 0:
                            arxiv_id = id_parts[-1]
                            # Remove version if present
                            if 'v' in arxiv_id:
                                arxiv_id = arxiv_id.split('v')[0]
                    
                    # Extract publication date
                    published = getattr(entry, 'published', '')
                    year = ""
                    try:
                        if published:
                            # Handle different date formats
                            if 'T' in published:
                                pub_date = datetime.strptime(published, "%Y-%m-%dT%H:%M:%SZ")
                            else:
                                # Try other common formats
                                for fmt in ["%Y-%m-%d", "%d %b %Y", "%B %d, %Y"]:
                                    try:
                                        pub_date = datetime.strptime(published, fmt)
                                        break
                                    except ValueError:
                                        continue
                            year = str(pub_date.year)
                    except (ValueError, TypeError, NameError) as e:
                        logger.debug(f"Error parsing publication date '{published}': {e}")
                    
                    # Extract PDF URL
                    pdf_url = ""
                    if hasattr(entry, 'links'):
                        for link in entry.links:
                            if hasattr(link, 'title') and link.title == "pdf" and hasattr(link, 'href'):
                                pdf_url = link.href
                                break
                            elif hasattr(link, 'type') and link.type == "application/pdf" and hasattr(link, 'href'):
                                pdf_url = link.href
                                break
                    
                    # Extract primary category
                    primary_category = ""
                    if hasattr(entry, 'arxiv_primary_category'):
                        primary_category = getattr(entry.arxiv_primary_category, 'term', '')
                    elif categories:
                        primary_category = categories[0]  # Use first category as primary if none specified
                    
                    # Create result entry
                    result = {
                        "id": arxiv_id,
                        "title": getattr(entry, 'title', '').replace("\n", " ").strip(),
                        "authors": authors,
                        "journal": "arXiv",
                        "publication_date": published,
                        "year": year,
                        "abstract": getattr(entry, 'summary', '').replace("\n", " ").strip(),
                        "categories": categories,
                        "url": getattr(entry, 'link', ''),
                        "pdf_url": pdf_url,
                        "identifier": primary_category,
                        "source": "arxiv",
                        "provider": "arxiv"
                    }
                    
                    results.append(result)
                    
                except Exception as entry_error:
                    # Log error for this specific entry but continue processing others
                    logger.warning(f"Error processing ArXiv entry: {entry_error}")
                    continue
            
            logger.info(f"Found {len(results)} results from ArXiv search")
            return results
            
        except Exception as e:
            logger.error(f"Error searching ArXiv: {str(e)}")
            # Log traceback for debugging
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return []
    
    async def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve detailed information about a specific ArXiv paper.
        
        Args:
            entity_id (str): ArXiv ID (e.g., 2101.12345)
            
        Returns:
            Optional[Dict[str, Any]]: Paper information or None if not found
        """
        if not self._is_initialized:
            await self.initialize()
            
        try:
            # Clean the entity_id in case it has the full URL
            if "/" in entity_id:
                entity_id = entity_id.split("/")[-1]
            
            # Some ArXiv IDs may include version (v1, v2, etc.)
            base_id = entity_id.split("v")[0] if "v" in entity_id else entity_id
            
            # Call ArXiv API for the specific paper
            url = f"{self.base_url}/query"
            params = {
                "id_list": base_id,
                "max_results": "1"
            }
                
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            
            # Parse response
            feed = parse_feed(response.text)
            
            # Check if we found the paper
            if not feed.entries:
                logger.warning(f"ArXiv paper {entity_id} not found")
                return None
                
            entry = feed.entries[0]
            
            # Extract authors
            authors = []
            for author in entry.get("authors", []):
                if "name" in author:
                    authors.append(author["name"])
            
            # Extract categories
            categories = []
            for category in entry.get("tags", []):
                if category.get("term"):
                    categories.append(category["term"])
            
            # Extract publication date
            published = entry.get("published", "")
            try:
                pub_date = datetime.strptime(published, "%Y-%m-%dT%H:%M:%SZ")
                year = pub_date.year
            except (ValueError, TypeError):
                year = ""
            
            # Create result
            result = {
                "id": entity_id,
                "title": entry.get("title", "").replace("\n", " ").strip(),
                "authors": authors,
                "journal": "arXiv",
                "publication_date": published,
                "year": str(year) if year else "",
                "abstract": entry.get("summary", "").replace("\n", " ").strip(),
                "categories": categories,
                "url": entry.get("link", ""),
                "pdf_url": next((link.href for link in entry.get("links", []) if link.get("title") == "pdf"), ""),
                "identifier": entry.get("arxiv_primary_category", {}).get("term", ""),
                "source": "arxiv",
                "provider": "arxiv"
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error fetching ArXiv paper {entity_id}: {str(e)}")
            return None
    
    async def get_related_entities(self, entity_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get papers related to the specified ArXiv paper.
        
        ArXiv doesn't provide a direct "related papers" API, so we implement this by:
        1. Getting the paper's categories and authors
        2. Searching for recent papers in the same categories by the same authors
        3. If that doesn't yield enough results, search for papers in the same categories with similar titles/abstracts
        
        Args:
            entity_id (str): ArXiv ID
            limit (int, optional): Maximum number of results. Defaults to 10.
            
        Returns:
            List[Dict[str, Any]]: List of related papers
        """
        if not self._is_initialized:
            await self.initialize()
            
        try:
            # First get the paper's details
            paper = await self.get_entity(entity_id)
            if not paper:
                return []
                
            # Extract categories and authors
            categories = paper.get("categories", [])
            authors = paper.get("authors", [])
            title_terms = ' '.join([term for term in paper.get("title", "").lower().split() if len(term) > 3])
            
            # Strategy 1: Search for papers by the same authors in the same categories
            author_results = []
            if authors:
                # Take up to 2 authors to avoid too narrow searches
                main_authors = authors[:2]
                author_query = " OR ".join([f"au:\"{author}\"" for author in main_authors])
                
                if categories:
                    main_category = categories[0]
                    search_query = f"({author_query}) AND cat:{main_category}"
                else:
                    search_query = author_query
                
                # Get papers by same authors
                author_results = await self.search(search_query, limit=limit)
                
                # Filter out the original paper
                author_results = [r for r in author_results if r["id"] != entity_id]
            
            # If we don't have enough results, search by categories and title similarity
            if len(author_results) < limit and categories:
                main_category = categories[0]
                
                # Create a search query using the most specific terms from the title
                title_terms = ' '.join([term for term in paper.get("title", "").lower().split() if len(term) > 3][:5])
                search_query = f"cat:{main_category} AND (ti:{title_terms} OR abs:{title_terms})"
                
                # Get papers in same category with similar title/abstract
                category_results = await self.search(search_query, limit=limit*2)
                
                # Filter out the original paper and those already in author_results
                existing_ids = {entity_id} | {r["id"] for r in author_results}
                category_results = [r for r in category_results if r["id"] not in existing_ids]
                
                # Combine results
                combined_results = author_results + category_results
                
                return combined_results[:limit]
            
            return author_results[:limit]
            
        except Exception as e:
            logger.error(f"Error finding related papers for {entity_id}: {str(e)}")
            return []
    
    def format_citation(self, entity: Dict[str, Any], style: str = "apa") -> str:
        """
        Format citation for an ArXiv paper according to the specified citation style.
        
        Args:
            entity (Dict[str, Any]): The paper to cite
            style (str, optional): Citation style (e.g., 'apa', 'mla', 'chicago', 'vancouver'). 
                Defaults to "apa".
                
        Returns:
            str: Formatted citation string
        """
        # Default to APA if style not supported
        if style.lower() not in self.citation_templates:
            style = "apa"
            
        # Extract data needed for citation
        authors = entity.get("authors", [])
        title = entity.get("title", "")
        year = entity.get("year", "")
        identifier = entity.get("identifier", "")
        url = entity.get("url", "")
        
        # Format authors according to style
        formatted_authors = ""
        if style == "apa":
            if len(authors) == 1:
                formatted_authors = authors[0]
            elif len(authors) == 2:
                formatted_authors = f"{authors[0]} & {authors[1]}"
            elif len(authors) > 2:
                formatted_authors = f"{authors[0]} et al."
        elif style == "mla":
            if len(authors) == 1:
                formatted_authors = authors[0]
            elif len(authors) == 2:
                formatted_authors = f"{authors[0]} and {authors[1]}"
            elif len(authors) > 2:
                formatted_authors = f"{authors[0]} et al."
        elif style == "chicago":
            if len(authors) == 1:
                formatted_authors = authors[0]
            elif len(authors) == 2:
                formatted_authors = f"{authors[0]} and {authors[1]}"
            elif len(authors) > 2:
                formatted_authors = f"{authors[0]} et al."
        elif style == "vancouver":
            if len(authors) <= 6:
                formatted_authors = ", ".join(authors)
            else:
                formatted_authors = ", ".join(authors[:6]) + ", et al."
                
        # Create dictionary with formatted components
        citation_data = {
            "authors": formatted_authors,
            "title": title,
            "journal": "arXiv preprint",
            "year": year,
            "identifier": f"arXiv:{entity.get('id', '')}" if entity.get('id') else "",
            "url": url
        }
        
        # Use the template to format the citation
        citation = self.citation_templates[style].format(**citation_data)
        
        return citation