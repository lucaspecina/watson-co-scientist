"""
PDF retrieval module for downloading scientific papers from various sources.
"""

import os
import logging
import httpx
import asyncio
import urllib.parse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger("co_scientist")

class PDFRetriever:
    """
    Retrieves scientific paper PDFs from URLs.
    
    This class handles downloading PDFs from various scientific repositories
    and publishers, with special handling for different sources.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the PDF retriever.
        
        Args:
            config (Dict[str, Any], optional): Configuration dictionary. Defaults to None.
        """
        self.config = config or {}
        
        # Set up PDF storage directory
        self.storage_dir = self.config.get('storage_dir', 'data/papers')
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # HTTP client for downloads
        self.client = httpx.AsyncClient(
            timeout=60.0,
            follow_redirects=True,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
        )
        
        # Map of publisher domains to specialized handling functions
        self.publisher_handlers = {
            "arxiv.org": self._handle_arxiv,
            "pubmed.ncbi.nlm.nih.gov": self._handle_pubmed,
            "www.nature.com": self._handle_nature,
            "science.org": self._handle_science,
            "www.sciencedirect.com": self._handle_sciencedirect,
            "link.springer.com": self._handle_springer,
        }
        
    async def download_pdf(self, url: str, paper_id: str = None, output_path: str = None) -> Optional[str]:
        """
        Download a PDF from a URL.
        
        Args:
            url (str): The URL to the PDF or paper landing page
            paper_id (str, optional): Unique identifier for the paper. Defaults to None.
            output_path (str, optional): Custom output path. Defaults to None.
            
        Returns:
            Optional[str]: Path to the downloaded PDF or None if download failed
        """
        try:
            # Parse URL to identify publisher
            parsed_url = urllib.parse.urlparse(url)
            domain = parsed_url.netloc
            
            # Generate paper_id from URL if not provided
            if not paper_id:
                paper_id = self._generate_paper_id(url)
            
            # Set output path
            if not output_path:
                output_path = os.path.join(self.storage_dir, f"{paper_id}.pdf")
            
            # Check if file already exists
            if os.path.exists(output_path):
                logger.info(f"PDF already exists at {output_path}")
                return output_path
            
            # Check if we have a specialized handler for this publisher
            if domain in self.publisher_handlers:
                pdf_url = await self.publisher_handlers[domain](url)
                if not pdf_url:
                    logger.warning(f"Could not determine PDF URL for {url}")
                    return None
            else:
                # If no specialized handler, assume the URL is directly to a PDF
                # or try to determine if it's a PDF by requesting headers
                if url.lower().endswith('.pdf'):
                    pdf_url = url
                else:
                    # Check if URL points to a PDF by examining content type
                    head_response = await self.client.head(url)
                    content_type = head_response.headers.get('Content-Type', '')
                    if 'application/pdf' in content_type.lower():
                        pdf_url = url
                    else:
                        logger.warning(f"URL does not appear to be a PDF: {url}")
                        return None
            
            # Download the PDF
            logger.info(f"Downloading PDF from {pdf_url} to {output_path}")
            response = await self.client.get(pdf_url)
            response.raise_for_status()
            
            # Verify the content is actually a PDF
            content_type = response.headers.get('Content-Type', '')
            if 'application/pdf' not in content_type.lower():
                # Some quick heuristic check for PDF signature
                if not response.content.startswith(b'%PDF-'):
                    logger.warning(f"Downloaded content is not a PDF from {pdf_url}")
                    return None
            
            # Save the PDF
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Successfully downloaded PDF to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error downloading PDF from {url}: {str(e)}")
            return None
    
    async def bulk_download(self, urls: List[Dict[str, str]], max_concurrent: int = 5) -> Dict[str, Optional[str]]:
        """
        Download multiple PDFs concurrently.
        
        Args:
            urls (List[Dict[str, str]]): List of dictionaries with 'url' and optional 'paper_id' keys
            max_concurrent (int, optional): Maximum number of concurrent downloads. Defaults to 5.
            
        Returns:
            Dict[str, Optional[str]]: Dictionary mapping URLs to downloaded file paths or None if failed
        """
        # Create semaphore to limit concurrent downloads
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def download_with_semaphore(url_dict):
            url = url_dict['url']
            paper_id = url_dict.get('paper_id')
            
            async with semaphore:
                path = await self.download_pdf(url, paper_id)
                return (url, path)
        
        # Create tasks for all downloads
        tasks = [download_with_semaphore(url_dict) for url_dict in urls]
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        download_results = {}
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error in bulk download: {str(result)}")
                continue
                
            url, path = result
            download_results[url] = path
        
        return download_results
    
    def _generate_paper_id(self, url: str) -> str:
        """Generate a unique paper ID from the URL."""
        # Try to extract a sensible ID from the URL
        parsed_url = urllib.parse.urlparse(url)
        path_parts = parsed_url.path.strip('/').split('/')
        
        # Handle different publisher URL patterns
        if 'arxiv.org' in parsed_url.netloc:
            # ArXiv pattern: https://arxiv.org/abs/1234.56789
            for part in path_parts:
                if part.isdigit() or '.' in part and any(c.isdigit() for c in part):
                    return f"arxiv_{part.replace('.', '_')}"
        elif 'doi.org' in parsed_url.netloc:
            # DOI pattern: https://doi.org/10.1234/journal.pone.0123456
            return f"doi_{'_'.join(path_parts)}"
        elif 'pubmed' in parsed_url.netloc:
            # PubMed pattern: https://pubmed.ncbi.nlm.nih.gov/12345678/
            for part in path_parts:
                if part.isdigit():
                    return f"pubmed_{part}"
        
        # Fallback: use domain + hash of path
        import hashlib
        path_hash = hashlib.md5(parsed_url.path.encode()).hexdigest()[:10]
        domain = parsed_url.netloc.replace('www.', '').split('.')[0]
        return f"{domain}_{path_hash}"
    
    async def _handle_arxiv(self, url: str) -> Optional[str]:
        """Handle ArXiv URLs to get direct PDF link."""
        # Convert abstract URL to PDF URL
        # Example: https://arxiv.org/abs/1234.56789 -> https://arxiv.org/pdf/1234.56789.pdf
        parsed_url = urllib.parse.urlparse(url)
        if '/abs/' in parsed_url.path:
            pdf_path = parsed_url.path.replace('/abs/', '/pdf/') + '.pdf'
            return f"{parsed_url.scheme}://{parsed_url.netloc}{pdf_path}"
        elif '/pdf/' in parsed_url.path:
            if not url.lower().endswith('.pdf'):
                return f"{url}.pdf"
            return url
        else:
            logger.warning(f"Unrecognized ArXiv URL pattern: {url}")
            return None
    
    async def _handle_pubmed(self, url: str) -> Optional[str]:
        """Handle PubMed URLs to find PDF link."""
        # PubMed doesn't host PDFs directly, but we can try to find links to publisher sites
        try:
            response = await self.client.get(url)
            response.raise_for_status()
            
            # Look for PDF links in the HTML content
            html_content = response.text
            
            # Try to find common patterns for PDF links in PubMed pages
            pdf_indicators = [
                'href="https://www.ncbi.nlm.nih.gov/pmc/articles/',
                'class="link-item pmc"',
                'data-ga-action="PMC Article"'
            ]
            
            for indicator in pdf_indicators:
                if indicator in html_content:
                    # Found a PMC link, follow it to get the PDF
                    # Extract the PMC link
                    import re
                    pmc_match = re.search(r'href="(https://www\.ncbi\.nlm\.nih\.gov/pmc/articles/[^"]+)"', html_content)
                    if pmc_match:
                        pmc_url = pmc_match.group(1)
                        # Follow the PMC link to find the PDF
                        logger.info(f"Following PMC link: {pmc_url}")
                        return await self._handle_pmc(pmc_url)
            
            logger.warning(f"Could not find PDF link on PubMed page: {url}")
            return None
            
        except Exception as e:
            logger.error(f"Error handling PubMed URL {url}: {str(e)}")
            return None
    
    async def _handle_pmc(self, url: str) -> Optional[str]:
        """Handle PubMed Central (PMC) URLs to find PDF link."""
        try:
            response = await self.client.get(url)
            response.raise_for_status()
            
            # Look for PDF links in the HTML content
            html_content = response.text
            
            # Try to find the PDF link
            import re
            pdf_match = re.search(r'href="([^"]+\.pdf)"', html_content)
            if pdf_match:
                pdf_url = pdf_match.group(1)
                # Handle relative URLs
                if pdf_url.startswith('/'):
                    parsed_url = urllib.parse.urlparse(url)
                    pdf_url = f"{parsed_url.scheme}://{parsed_url.netloc}{pdf_url}"
                return pdf_url
            
            logger.warning(f"Could not find PDF link on PMC page: {url}")
            return None
            
        except Exception as e:
            logger.error(f"Error handling PMC URL {url}: {str(e)}")
            return None
    
    async def _handle_nature(self, url: str) -> Optional[str]:
        """Handle Nature journal URLs to find PDF link."""
        try:
            # Nature typically offers PDF downloads via a predictable URL pattern
            # Example: https://www.nature.com/articles/s41586-021-03819-2 -> 
            #          https://www.nature.com/articles/s41586-021-03819-2.pdf
            if url.endswith('.pdf'):
                return url
            else:
                return f"{url}.pdf"
                
        except Exception as e:
            logger.error(f"Error handling Nature URL {url}: {str(e)}")
            return None
    
    async def _handle_science(self, url: str) -> Optional[str]:
        """Handle Science journal URLs to find PDF link."""
        try:
            # For Science, we need to find the PDF link on the page
            response = await self.client.get(url)
            response.raise_for_status()
            
            html_content = response.text
            
            # Look for PDF link
            import re
            pdf_match = re.search(r'href="([^"]+\.full\.pdf)"', html_content)
            if pdf_match:
                pdf_url = pdf_match.group(1)
                # Handle relative URLs
                if pdf_url.startswith('/'):
                    parsed_url = urllib.parse.urlparse(url)
                    pdf_url = f"{parsed_url.scheme}://{parsed_url.netloc}{pdf_url}"
                return pdf_url
            
            logger.warning(f"Could not find PDF link on Science page: {url}")
            return None
            
        except Exception as e:
            logger.error(f"Error handling Science URL {url}: {str(e)}")
            return None
    
    async def _handle_sciencedirect(self, url: str) -> Optional[str]:
        """Handle ScienceDirect URLs to find PDF link."""
        try:
            # ScienceDirect uses a special PDF URL format
            response = await self.client.get(url)
            response.raise_for_status()
            
            html_content = response.text
            
            # Look for PDF link
            import re
            pdf_match = re.search(r'pdfUrl":"([^"]+)"', html_content)
            if pdf_match:
                pdf_url = pdf_match.group(1).replace('\\', '')
                # Handle absolute vs relative URLs
                if pdf_url.startswith('http'):
                    return pdf_url
                else:
                    parsed_url = urllib.parse.urlparse(url)
                    return f"{parsed_url.scheme}://{parsed_url.netloc}{pdf_url}"
            
            logger.warning(f"Could not find PDF link on ScienceDirect page: {url}")
            return None
            
        except Exception as e:
            logger.error(f"Error handling ScienceDirect URL {url}: {str(e)}")
            return None
    
    async def _handle_springer(self, url: str) -> Optional[str]:
        """Handle Springer URLs to find PDF link."""
        try:
            # Springer journals typically use a /content/pdf/ path
            response = await self.client.get(url)
            response.raise_for_status()
            
            html_content = response.text
            
            # Look for PDF link
            import re
            pdf_match = re.search(r'href="([^"]+/pdf/[^"]+\.pdf)"', html_content)
            if pdf_match:
                pdf_url = pdf_match.group(1)
                # Handle protocol-relative URLs
                if pdf_url.startswith('//'):
                    parsed_url = urllib.parse.urlparse(url)
                    pdf_url = f"{parsed_url.scheme}:{pdf_url}"
                # Handle relative URLs
                elif pdf_url.startswith('/'):
                    parsed_url = urllib.parse.urlparse(url)
                    pdf_url = f"{parsed_url.scheme}://{parsed_url.netloc}{pdf_url}"
                return pdf_url
            
            logger.warning(f"Could not find PDF link on Springer page: {url}")
            return None
            
        except Exception as e:
            logger.error(f"Error handling Springer URL {url}: {str(e)}")
            return None
            
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()