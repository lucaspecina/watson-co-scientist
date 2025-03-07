"""
Web search functionality for the Co-Scientist system.
This module provides tools for performing scientific literature searches and retrieving content.
"""

import os
import json
import logging
import asyncio
from typing import List, Dict, Any, Optional, Union
import httpx
from bs4 import BeautifulSoup
import requests

# Import the official Tavily client if available
try:
    from tavily import TavilyClient, AsyncTavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False
    print("Tavily Python client not available. Will use HTTP requests directly.")

logger = logging.getLogger("co_scientist")

class WebSearchTool:
    """A tool for performing web searches with multiple provider options."""
    
    def __init__(self, api_key: Optional[str] = None, provider: str = "tavily"):
        """
        Initialize the web search tool.
        
        Args:
            api_key (Optional[str]): The API key for the search provider.
            provider (str): The search provider to use ('tavily', 'bing', or 'serper').
        """
        self.provider = provider.lower()
        
        # Load appropriate API key based on provider
        if self.provider == "tavily":
            self.api_key = api_key or os.environ.get("TAVILY_API_KEY")
            self.base_url = "https://api.tavily.com/search"
            # Print Tavily API key info for debugging
            if self.api_key:
                key_preview = f"{self.api_key[:5]}...{self.api_key[-5:]}" if len(self.api_key) > 10 else self.api_key
                print(f"Loaded Tavily API key: {key_preview}")
            else:
                print("No Tavily API key found!")
        elif self.provider == "bing":
            self.api_key = api_key or os.environ.get("BING_SEARCH_API_KEY")
            self.base_url = "https://api.bing.microsoft.com/v7.0/search"
        elif self.provider == "serper":
            self.api_key = api_key or os.environ.get("SERPER_API_KEY")
            self.base_url = "https://serpapi.com/search"
        else:
            raise ValueError(f"Unsupported search provider: {provider}")
            
        if not self.api_key:
            logger.warning(f"No API key provided for {provider} search. Web search will be disabled.")
            
        # Initialize HTTP client
        self.client = httpx.AsyncClient(timeout=30.0)
        
    async def search(self, query: str, count: int = 10, search_type: str = "search") -> List[Dict[str, Any]]:
        """
        Perform a web search.
        
        Args:
            query (str): The search query.
            count (int): The number of results to return.
            search_type (str): Type of search to perform (general, scientific, news)
            
        Returns:
            List[Dict[str, Any]]: The search results.
        """
        if not self.api_key:
            logger.warning(f"Web search is disabled. No API key provided for {self.provider}.")
            return []
            
        try:
            if self.provider == "tavily":
                return await self._tavily_search(query, count, search_type)
            elif self.provider == "bing":
                return await self._bing_search(query, count)
            elif self.provider == "serper":
                return await self._serper_search(query, count)
                
        except Exception as e:
            logger.error(f"Error performing web search with {self.provider}: {e}")
            return []
            
    async def _tavily_search(self, query: str, count: int, search_type: str) -> List[Dict[str, Any]]:
        """Perform a search using the Tavily API."""
        # Make sure we output what's happening
        print(f"Calling Tavily API with query: '{query}'")
        key_preview = f"{self.api_key[:5]}...{self.api_key[-5:]}" if len(self.api_key) > 10 else self.api_key
        print(f"Using API key: {key_preview}")
        
        search_types_map = {
            "scientific": "scientific",
            "academic": "scientific",
            "news": "news",
            "search": "search"
        }
        
        # Use scientific search type if available
        tavily_search_type = search_types_map.get(search_type.lower(), "search")
        
        try:
            # Check if the Tavily client is available
            if TAVILY_AVAILABLE:
                print("Using official Tavily AsyncClient")
                # Use the official Async Tavily client
                client = AsyncTavilyClient(api_key=self.api_key)
                
                # Prepare search parameters
                search_params = {
                    "query": query,
                    "search_depth": "advanced",
                    "max_results": count,
                    "include_answer": True,
                }
                
                # Add search_type if it's specified
                if tavily_search_type != "search":
                    search_params["search_type"] = tavily_search_type
                
                # Make an async search
                try:
                    data = await client.search(**search_params)
                    print(f"Tavily API search completed successfully with AsyncClient")
                except Exception as e:
                    print(f"Error with AsyncTavilyClient: {str(e)}")
                    print("Falling back to synchronous TavilyClient...")
                    
                    # Fall back to sync client if async fails
                    sync_client = TavilyClient(api_key=self.api_key)
                    loop = asyncio.get_event_loop()
                    data = await loop.run_in_executor(
                        None, 
                        lambda: sync_client.search(**search_params)
                    )
                    print(f"Tavily API search completed successfully with sync client")
                
            else:
                print("Using direct HTTP request to Tavily API")
                # Fall back to direct HTTP requests
                headers = {
                    "Content-Type": "application/json",
                    "X-API-Key": self.api_key  # Try this key format instead
                }
                
                payload = {
                    "query": query,
                    "search_depth": "advanced",
                    "max_results": count,
                    "include_answer": True,
                }
                
                # Add search_type if it's specified
                if tavily_search_type != "search":
                    payload["search_type"] = tavily_search_type
                
                # Add include_domains only for scientific search
                if tavily_search_type == "scientific":
                    payload["include_domains"] = [
                        "scholar.google.com", "pubmed.ncbi.nlm.nih.gov", "nature.com", 
                        "science.org", "sciencedirect.com", "ncbi.nlm.nih.gov", "arxiv.org"
                    ]
                
                # Use requests for simplicity (non-async but in executor)
                def make_request():
                    response = requests.post(
                        "https://api.tavily.com/search", 
                        headers=headers, 
                        json=payload, 
                        timeout=60.0
                    )
                    response.raise_for_status()
                    return response.json()
                
                # Run in executor to avoid blocking
                loop = asyncio.get_event_loop()
                data = await loop.run_in_executor(None, make_request)
            
            # Print a sample of the data to debug
            print(f"Tavily API response: {str(data)[:200]}...")
            
            # Extract results from Tavily format
            results = []
            if "results" in data:
                for item in data["results"]:
                    results.append({
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "snippet": item.get("content", ""),
                        "source": "tavily",
                        "publication_date": item.get("published_date", ""),
                        "domain": item.get("domain", "")
                    })
                    
            # Add the generated answer if available
            if "answer" in data and data["answer"]:
                results.insert(0, {
                    "title": "Tavily Generated Answer",
                    "url": "",
                    "snippet": data["answer"],
                    "source": "tavily_answer",
                    "is_generated": True
                })
                
            logger.info(f"Performed Tavily {tavily_search_type} search for '{query}' and got {len(results)} results")
            return results
            
        except Exception as e:
            print(f"Error in Tavily search: {str(e)}")
            logger.error(f"Error in Tavily search: {str(e)}")
            # Return empty results in case of error
            return []
            
    async def _bing_search(self, query: str, count: int) -> List[Dict[str, Any]]:
        """Perform a search using the Bing API."""
        headers = {"Ocp-Apim-Subscription-Key": self.api_key}
        params = {"q": query, "count": count, "responseFilter": "Webpages"}
        
        response = await self.client.get(self.base_url, headers=headers, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        # Extract the relevant information from the response
        results = []
        if "webPages" in data and "value" in data["webPages"]:
            for item in data["webPages"]["value"]:
                results.append({
                    "title": item.get("name", ""),
                    "url": item.get("url", ""),
                    "snippet": item.get("snippet", ""),
                    "source": "bing"
                })
        
        logger.info(f"Performed Bing search for '{query}' and got {len(results)} results")
        return results
        
    async def _serper_search(self, query: str, count: int) -> List[Dict[str, Any]]:
        """Perform a search using the SerperAPI."""
        headers = {"X-API-KEY": self.api_key}
        params = {"q": query, "num": count}
        
        response = await self.client.get(self.base_url, headers=headers, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        # Extract the relevant information from the response
        results = []
        if "organic" in data:
            for item in data["organic"]:
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                    "source": "serper"
                })
        
        logger.info(f"Performed Serper search for '{query}' and got {len(results)} results")
        return results
            
    async def fetch_content(self, url: str) -> Optional[str]:
        """
        Fetch the content of a web page and extract main text.
        
        Args:
            url (str): The URL to fetch.
            
        Returns:
            Optional[str]: The extracted main content of the page, or None if there was an error.
        """
        # Check if the URL is empty or missing protocol
        if not url or not (url.startswith('http://') or url.startswith('https://')):
            print(f"Error fetching content from {url}: Request URL is missing an 'http://' or 'https://' protocol.")
            return None
            
        try:
            # Add browser-like headers to avoid 403 errors
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
                "Cache-Control": "max-age=0"
            }
            
            # Check if we're accessing a scientific site that might need special handling
            scientific_domains = ["ncbi.nlm.nih.gov", "pubmed", "science.org", "nature.com", "frontiersin.org", "sciencedirect.com"]
            is_scientific_site = any(domain in url for domain in scientific_domains)
            
            if is_scientific_site:
                # For scientific sites, use a simplified approach to reduce chance of 403 errors
                # In a real implementation, we would use site-specific APIs instead
                print(f"Detected scientific site: {url} - Using simplified content extraction")
                # For scientific sites, just return a placeholder since we already have the abstract from search
                return f"[Content from scientific article: {url}. Full text access requires authentication or API access]"
            
            # For regular sites, proceed with normal extraction
            response = await self.client.get(url, headers=headers, follow_redirects=True)
            response.raise_for_status()
            
            # Get the content
            html_content = response.text
            
            # Parse HTML and extract text
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "header", "footer", "nav"]):
                script.extract()
                
            # Get text
            text = soup.get_text(separator='\n', strip=True)
            
            # Remove extra whitespace
            text = '\n'.join([line.strip() for line in text.split('\n') if line.strip()])
            
            # Limit the length to avoid huge content
            if len(text) > 50000:
                text = text[:50000] + "... [content truncated]"
            
            logger.info(f"Fetched and parsed content from {url} ({len(text)} chars)")
            return text
            
        except Exception as e:
            print(f"Error fetching content from {url}: {e}")
            logger.error(f"Error fetching content from {url}: {e}")
            return None
            
    async def search_and_fetch(self, query: str, max_results: int = 5, search_type: str = "scientific") -> List[Dict[str, Any]]:
        """
        Perform a web search and fetch the content of the top results.
        
        Args:
            query (str): The search query.
            max_results (int): The maximum number of results to fetch.
            search_type (str): Type of search (scientific, news, search)
            
        Returns:
            List[Dict[str, Any]]: The search results with content.
        """
        # Perform the search
        results = await self.search(query, count=max_results, search_type=search_type)
        
        # Track which results have content
        results_with_content = []
        
        # Process each result
        for i, result in enumerate(results):
            # For tavily_answer or any result with is_generated=True, use snippet as content
            if result.get("source") == "tavily_answer" or result.get("is_generated", False):
                result["content"] = result["snippet"]
                results_with_content.append(result)
                continue
                
            # For results with URLs, fetch content
            if result.get("url"):
                content = await self.fetch_content(result["url"])
                if content:
                    result["content"] = content
                    results_with_content.append(result)
                else:
                    # If fetching fails but we have a snippet, use that as content
                    print(f"Using snippet as content for {result.get('title')}")
                    result["content"] = result.get("snippet", "No content available")
                    results_with_content.append(result)
            else:
                # For results without URL but with snippet, use snippet as content
                if result.get("snippet"):
                    result["content"] = result["snippet"]
                    results_with_content.append(result)
                    
        # Return all results with content (should be all results now)
        print(f"Retrieved {len(results_with_content)} results with content out of {len(results)} total results")
        return results_with_content

class ScientificLiteratureSearch:
    """A specialized tool for scientific literature searching."""
    
    def __init__(self, api_key: Optional[str] = None, provider: str = "tavily"):
        """
        Initialize the scientific literature search tool.
        
        Args:
            api_key (Optional[str]): The API key for the search provider.
            provider (str): The search provider to use.
        """
        self.web_search = WebSearchTool(api_key=api_key, provider=provider)
        
    async def search_literature(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search for scientific literature.
        
        Args:
            query (str): The search query.
            max_results (int): The maximum number of results to return.
            
        Returns:
            List[Dict[str, Any]]: The search results with content.
        """
        # Add scientific terms to the query
        scientific_query = f"{query} scientific research paper peer-reviewed"
        
        return await self.web_search.search_and_fetch(
            scientific_query,
            max_results=max_results,
            search_type="scientific"
        )
        
    async def search_with_citations(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Search for scientific literature and extract citations.
        
        Args:
            query (str): The search query.
            max_results (int): The maximum number of results to return.
            
        Returns:
            Dict[str, Any]: Search results and extracted citations.
        """
        results = await self.search_literature(query, max_results)
        
        # Extract and format citations
        citations = []
        for idx, result in enumerate(results):
            citation = {
                "id": f"ref_{idx + 1}",
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "authors": [],  # Would need more processing to extract author names
                "publication_date": result.get("publication_date", ""),
                "source": result.get("source", ""),
                "snippet": result.get("snippet", "")[:200] + "..." if result.get("snippet") else ""
            }
            citations.append(citation)
            
        return {
            "results": results,
            "citations": citations,
            "summary": f"Found {len(results)} scientific papers related to: {query}"
        }