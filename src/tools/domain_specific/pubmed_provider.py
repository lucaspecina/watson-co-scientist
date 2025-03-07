"""
PubMed knowledge provider for biomedical domain knowledge.
"""

import os
import logging
import httpx
import asyncio
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

from .base_provider import DomainKnowledgeProvider

logger = logging.getLogger("co_scientist")

class PubMedProvider(DomainKnowledgeProvider):
    """
    PubMed knowledge provider for biomedical domain knowledge.
    
    This provider interfaces with the NCBI E-utilities API to search and retrieve
    data from PubMed, a database of biomedical literature.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the PubMed provider.
        
        Args:
            config (Dict[str, Any], optional): Configuration for the provider. 
                May include 'api_key' for NCBI API. Defaults to None.
        """
        super().__init__(domain="biomedicine", config=config)
        
        # NCBI E-utilities base URL
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        
        # Get API key from config or environment
        self.api_key = self.config.get("api_key") or os.environ.get("NCBI_API_KEY", "")
        
        # HTTP client for API requests
        self.client = httpx.AsyncClient(timeout=30.0)
        
        # Citation format templates
        self.citation_templates = {
            "apa": "{authors} ({year}). {title}. {journal}, {volume}({issue}), {pages}. {doi}",
            "mla": "{authors}. \"{title}.\" {journal}, vol. {volume}, no. {issue}, {year}, pp. {pages}. {doi}",
            "chicago": "{authors}. \"{title}.\" {journal} {volume}, no. {issue} ({year}): {pages}. {doi}",
            "vancouver": "{authors}. {title}. {journal}. {year};{volume}({issue}):{pages}. {doi}"
        }
    
    async def initialize(self) -> bool:
        """
        Initialize the PubMed provider.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        try:
            # Test connection to NCBI E-utilities
            params = {"db": "pubmed", "term": "test", "retmax": "1", "retmode": "json"}
            if self.api_key:
                params["api_key"] = self.api_key
                
            url = f"{self.base_url}/esearch.fcgi"
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            
            self._is_initialized = True
            logger.info("PubMed provider initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize PubMed provider: {str(e)}")
            return False
    
    async def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search PubMed for biomedical literature.
        
        Args:
            query (str): The search query
            limit (int, optional): Maximum number of results. Defaults to 10.
            
        Returns:
            List[Dict[str, Any]]: List of search results
        """
        if not self._is_initialized:
            await self.initialize()
            
        try:
            # Step 1: Use ESearch to get PMIDs
            search_url = f"{self.base_url}/esearch.fcgi"
            search_params = {
                "db": "pubmed",
                "term": query,
                "retmax": str(limit),
                "retmode": "json",
                "sort": "relevance"
            }
            
            if self.api_key:
                search_params["api_key"] = self.api_key
                
            search_response = await self.client.get(search_url, params=search_params)
            search_response.raise_for_status()
            search_data = search_response.json()
            
            # Extract PMIDs
            pmids = search_data.get("esearchresult", {}).get("idlist", [])
            if not pmids:
                return []
                
            # Step 2: Use ESummary to get article details
            summary_url = f"{self.base_url}/esummary.fcgi"
            summary_params = {
                "db": "pubmed",
                "id": ",".join(pmids),
                "retmode": "json"
            }
            
            if self.api_key:
                summary_params["api_key"] = self.api_key
                
            summary_response = await self.client.get(summary_url, params=summary_params)
            summary_response.raise_for_status()
            summary_data = summary_response.json()
            
            # Process results
            results = []
            for pmid in pmids:
                article = summary_data.get("result", {}).get(pmid, {})
                if not article:
                    continue
                    
                # Extract authors
                authors = []
                for author in article.get("authors", []):
                    if author.get("name", "") != "":
                        authors.append(author["name"])
                
                # Create result entry
                result = {
                    "id": pmid,
                    "title": article.get("title", ""),
                    "authors": authors,
                    "journal": article.get("fulljournalname", article.get("source", "")),
                    "publication_date": article.get("pubdate", ""),
                    "year": article.get("pubdate", "").split()[0] if article.get("pubdate") else "",
                    "volume": article.get("volume", ""),
                    "issue": article.get("issue", ""),
                    "pages": article.get("pages", ""),
                    "doi": article.get("elocationid", ""),
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    "abstract": "",  # Abstract not included in summary, requires separate fetch
                    "source": "pubmed",
                    "provider": "pubmed"
                }
                
                results.append(result)
                
            # If we need abstracts, fetch them (but this is an additional API call)
            # In a production system, you might want to fetch abstracts only when explicitly requested
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching PubMed: {str(e)}")
            return []
    
    async def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve detailed information about a specific PubMed article.
        
        Args:
            entity_id (str): PubMed ID (PMID)
            
        Returns:
            Optional[Dict[str, Any]]: Article information or None if not found
        """
        if not self._is_initialized:
            await self.initialize()
            
        try:
            # Use EFetch to get full article details
            fetch_url = f"{self.base_url}/efetch.fcgi"
            fetch_params = {
                "db": "pubmed",
                "id": entity_id,
                "retmode": "xml"
            }
            
            if self.api_key:
                fetch_params["api_key"] = self.api_key
                
            fetch_response = await self.client.get(fetch_url, params=fetch_params)
            fetch_response.raise_for_status()
            
            # Parse XML response
            root = ET.fromstring(fetch_response.text)
            
            # Extract article details
            article = root.find(".//PubmedArticle/MedlineCitation/Article")
            if article is None:
                return None
                
            # Extract basic info
            title_element = article.find("ArticleTitle")
            title = title_element.text if title_element is not None else ""
            
            # Extract journal info
            journal = article.find("Journal")
            journal_title = ""
            if journal is not None:
                journal_title_element = journal.find("Title")
                journal_title = journal_title_element.text if journal_title_element is not None else ""
            
            # Extract authors
            authors = []
            author_list = article.find("AuthorList")
            if author_list is not None:
                for author_elem in author_list.findall("Author"):
                    last_name = author_elem.find("LastName")
                    fore_name = author_elem.find("ForeName")
                    
                    last = last_name.text if last_name is not None else ""
                    fore = fore_name.text if fore_name is not None else ""
                    
                    if last or fore:
                        authors.append(f"{last}, {fore}" if fore else last)
            
            # Extract abstract
            abstract_text = ""
            abstract = article.find("Abstract")
            if abstract is not None:
                abstract_parts = []
                for abstract_elem in abstract.findall("AbstractText"):
                    if abstract_elem.text:
                        abstract_parts.append(abstract_elem.text)
                abstract_text = " ".join(abstract_parts)
            
            # Create result
            result = {
                "id": entity_id,
                "title": title,
                "authors": authors,
                "journal": journal_title,
                "abstract": abstract_text,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{entity_id}/",
                "source": "pubmed",
                "provider": "pubmed"
            }
            
            # Extract additional metadata if available
            pub_date = root.find(".//PubmedArticle/MedlineCitation/Article/Journal/JournalIssue/PubDate")
            if pub_date is not None:
                year = pub_date.find("Year")
                result["year"] = year.text if year is not None else ""
                
                month = pub_date.find("Month")
                result["month"] = month.text if month is not None else ""
                
                day = pub_date.find("Day")
                result["day"] = day.text if day is not None else ""
                
                result["publication_date"] = " ".join(filter(None, [result.get("year", ""), 
                                                                 result.get("month", ""), 
                                                                 result.get("day", "")]))
            
            # Extract DOI if available
            article_id_list = root.find(".//PubmedArticle/PubmedData/ArticleIdList")
            if article_id_list is not None:
                for id_elem in article_id_list.findall("ArticleId"):
                    if id_elem.get("IdType") == "doi":
                        result["doi"] = id_elem.text
            
            return result
            
        except Exception as e:
            logger.error(f"Error fetching PubMed article {entity_id}: {str(e)}")
            return None
    
    async def get_related_entities(self, entity_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get articles related to the specified PubMed article.
        
        Args:
            entity_id (str): PubMed ID (PMID)
            limit (int, optional): Maximum number of results. Defaults to 10.
            
        Returns:
            List[Dict[str, Any]]: List of related articles
        """
        if not self._is_initialized:
            await self.initialize()
            
        try:
            # Use ELink to find related articles
            link_url = f"{self.base_url}/elink.fcgi"
            link_params = {
                "dbfrom": "pubmed",
                "db": "pubmed",
                "id": entity_id,
                "linkname": "pubmed_pubmed",
                "retmode": "json"
            }
            
            if self.api_key:
                link_params["api_key"] = self.api_key
                
            link_response = await self.client.get(link_url, params=link_params)
            link_response.raise_for_status()
            link_data = link_response.json()
            
            # Extract related PMIDs
            related_pmids = []
            try:
                linksets = link_data.get("linksets", [])
                if linksets and "linksetdbs" in linksets[0]:
                    for linksetdb in linksets[0]["linksetdbs"]:
                        if linksetdb.get("linkname") == "pubmed_pubmed":
                            related_pmids = linksetdb.get("links", [])
                            break
            except Exception as e:
                logger.warning(f"Error parsing related PMIDs: {str(e)}")
                
            # Limit the number of PMIDs
            related_pmids = related_pmids[:limit]
            
            if not related_pmids:
                return []
                
            # Convert the list of PMIDs to a comma-separated string
            pmids_str = ",".join(map(str, related_pmids))
            
            # Use ESummary to get article details
            summary_url = f"{self.base_url}/esummary.fcgi"
            summary_params = {
                "db": "pubmed",
                "id": pmids_str,
                "retmode": "json"
            }
            
            if self.api_key:
                summary_params["api_key"] = self.api_key
                
            summary_response = await self.client.get(summary_url, params=summary_params)
            summary_response.raise_for_status()
            summary_data = summary_response.json()
            
            # Process results
            results = []
            for pmid in related_pmids:
                pmid_str = str(pmid)
                article = summary_data.get("result", {}).get(pmid_str, {})
                if not article:
                    continue
                    
                # Extract authors
                authors = []
                for author in article.get("authors", []):
                    if author.get("name", "") != "":
                        authors.append(author["name"])
                
                # Create result entry
                result = {
                    "id": pmid_str,
                    "title": article.get("title", ""),
                    "authors": authors,
                    "journal": article.get("fulljournalname", article.get("source", "")),
                    "publication_date": article.get("pubdate", ""),
                    "year": article.get("pubdate", "").split()[0] if article.get("pubdate") else "",
                    "volume": article.get("volume", ""),
                    "issue": article.get("issue", ""),
                    "pages": article.get("pages", ""),
                    "doi": article.get("elocationid", ""),
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid_str}/",
                    "source": "pubmed",
                    "provider": "pubmed",
                    "related_to": entity_id
                }
                
                results.append(result)
                
            return results
            
        except Exception as e:
            logger.error(f"Error fetching related articles for {entity_id}: {str(e)}")
            return []
    
    def format_citation(self, entity: Dict[str, Any], style: str = "apa") -> str:
        """
        Format citation for a PubMed article according to the specified citation style.
        
        Args:
            entity (Dict[str, Any]): The article to cite
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
        journal = entity.get("journal", "")
        year = entity.get("year", "")
        volume = entity.get("volume", "")
        issue = entity.get("issue", "")
        pages = entity.get("pages", "")
        doi = entity.get("doi", "")
        
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
            if len(authors) == 1:
                formatted_authors = authors[0]
            elif len(authors) == 2:
                formatted_authors = f"{authors[0]}, {authors[1]}"
            elif len(authors) > 2:
                formatted_authors = f"{authors[0]} et al."
                
        # Format DOI
        formatted_doi = ""
        if doi:
            if style == "apa":
                formatted_doi = f"https://doi.org/{doi}"
            elif style == "mla":
                formatted_doi = f"doi:{doi}"
            elif style == "chicago":
                formatted_doi = f"https://doi.org/{doi}"
            elif style == "vancouver":
                formatted_doi = f"doi:{doi}"
                
        # Create dictionary with formatted components
        citation_data = {
            "authors": formatted_authors,
            "title": title,
            "journal": journal,
            "year": year,
            "volume": volume,
            "issue": issue,
            "pages": pages,
            "doi": formatted_doi
        }
        
        # Use the template to format the citation
        citation = self.citation_templates[style].format(**citation_data)
        
        return citation