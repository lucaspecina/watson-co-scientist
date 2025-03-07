"""
Web search functionality for the Co-Scientist system.
"""

import os
import json
import logging
import asyncio
from typing import List, Dict, Any, Optional
import httpx

logger = logging.getLogger("co_scientist")

class WebSearchTool:
    """A tool for performing web searches."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the web search tool.
        
        Args:
            api_key (Optional[str]): The API key for the search provider.
        """
        self.api_key = api_key or os.environ.get("BING_SEARCH_API_KEY")
        
        if not self.api_key:
            logger.warning("No API key provided for web search. Web search will be disabled.")
            
        # Initialize HTTP client
        self.client = httpx.AsyncClient(timeout=30.0)
        
    async def search(self, query: str, count: int = 10) -> List[Dict[str, Any]]:
        """
        Perform a web search.
        
        Args:
            query (str): The search query.
            count (int): The number of results to return.
            
        Returns:
            List[Dict[str, Any]]: The search results.
        """
        if not self.api_key:
            logger.warning("Web search is disabled. No API key provided.")
            return []
            
        try:
            # Use Bing Search API
            url = "https://api.bing.microsoft.com/v7.0/search"
            headers = {"Ocp-Apim-Subscription-Key": self.api_key}
            params = {"q": query, "count": count, "responseFilter": "Webpages"}
            
            response = await self.client.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract the relevant information from the response
            results = []
            if "webPages" in data and "value" in data["webPages"]:
                for item in data["webPages"]["value"]:
                    results.append({
                        "title": item.get("name", ""),
                        "url": item.get("url", ""),
                        "snippet": item.get("snippet", "")
                    })
            
            logger.info(f"Performed web search for '{query}' and got {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error performing web search: {e}")
            return []
            
    async def fetch_content(self, url: str) -> Optional[str]:
        """
        Fetch the content of a web page.
        
        Args:
            url (str): The URL to fetch.
            
        Returns:
            Optional[str]: The content of the page, or None if there was an error.
        """
        try:
            response = await self.client.get(url)
            response.raise_for_status()
            
            # Get the content
            content = response.text
            
            # For simplicity, we'll just return the raw HTML for now
            # In a real implementation, we would parse the HTML and extract the main text
            logger.info(f"Fetched content from {url} ({len(content)} bytes)")
            return content
            
        except Exception as e:
            logger.error(f"Error fetching content from {url}: {e}")
            return None
            
    async def search_and_fetch(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Perform a web search and fetch the content of the top results.
        
        Args:
            query (str): The search query.
            max_results (int): The maximum number of results to fetch.
            
        Returns:
            List[Dict[str, Any]]: The search results with content.
        """
        # Perform the search
        results = await self.search(query, count=max_results)
        
        # Fetch the content of each result
        for result in results:
            content = await self.fetch_content(result["url"])
            if content:
                result["content"] = content
                
        # Return only the results with content
        return [r for r in results if "content" in r]