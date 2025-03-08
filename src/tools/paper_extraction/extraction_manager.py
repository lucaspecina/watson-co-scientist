"""
Paper extraction manager for coordinating PDF retrieval, processing, and knowledge extraction.
"""

import os
import json
import asyncio
import logging
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from datetime import datetime

from .pdf_retriever import PDFRetriever
from .pdf_processor import PDFProcessor, ExtractedPaper
from .knowledge_extractor import KnowledgeExtractor

logger = logging.getLogger("co_scientist")

class PaperExtractionManager:
    """
    Manager for coordinating paper extraction, processing, and knowledge extraction.
    
    This class provides a unified interface for retrieving PDFs, extracting their
    content, and converting that content into structured knowledge.
    """
    
    def __init__(self, config: Dict[str, Any] = None, llm_provider=None):
        """
        Initialize the paper extraction manager.
        
        Args:
            config (Dict[str, Any], optional): Configuration dictionary. Defaults to None.
            llm_provider: LLM provider for semantic extraction. Defaults to None.
        """
        self.config = config or {}
        self.llm_provider = llm_provider
        
        # Configure directories
        base_dir = self.config.get('base_dir', 'data')
        self.pdf_dir = os.path.join(base_dir, 'papers')
        self.extraction_dir = os.path.join(base_dir, 'extractions')
        self.knowledge_dir = os.path.join(base_dir, 'knowledge')
        self.figure_dir = os.path.join(base_dir, 'figures')
        
        # Create directories
        os.makedirs(self.pdf_dir, exist_ok=True)
        os.makedirs(self.extraction_dir, exist_ok=True)
        os.makedirs(self.knowledge_dir, exist_ok=True)
        os.makedirs(self.figure_dir, exist_ok=True)
        
        # Configure sub-components
        pdf_config = {
            'storage_dir': self.pdf_dir
        }
        processor_config = {
            'image_dir': self.figure_dir
        }
        extractor_config = {}
        
        # Initialize sub-components
        self.retriever = PDFRetriever(pdf_config)
        self.processor = PDFProcessor(processor_config)
        self.extractor = KnowledgeExtractor(extractor_config, llm_provider=llm_provider)
        
        # Cache for processed papers
        self.paper_cache = {}
        
    async def extract_from_url(self, url: str, paper_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract knowledge from a paper URL.
        
        This method:
        1. Downloads the PDF
        2. Processes the PDF to extract structured content
        3. Extracts knowledge from the structured content
        
        Args:
            url (str): URL to the paper (can be PDF or landing page)
            paper_id (Optional[str], optional): Custom paper ID. Defaults to None.
            
        Returns:
            Dict[str, Any]: Extracted knowledge
        """
        try:
            # Step 1: Download PDF
            logger.info(f"Downloading PDF from {url}")
            pdf_path = await self.retriever.download_pdf(url, paper_id)
            if not pdf_path:
                logger.error(f"Failed to download PDF from {url}")
                return {"error": f"Failed to download PDF from {url}"}
                
            # Generate paper ID if not provided
            if not paper_id:
                paper_id = os.path.basename(pdf_path).replace('.pdf', '')
            
            # Step 2: Process PDF
            logger.info(f"Processing PDF {pdf_path}")
            extracted_paper = self.processor.process_pdf(pdf_path, self.figure_dir)
            
            # Save extraction results
            extraction_path = os.path.join(self.extraction_dir, f"{paper_id}_extracted.json")
            self.processor.save_extracted_paper(extracted_paper, extraction_path)
            logger.info(f"Saved extracted paper to {extraction_path}")
            
            # Cache the extracted paper
            self.paper_cache[paper_id] = extracted_paper
            
            # Step 3: Extract knowledge (if LLM provider available)
            if self.llm_provider:
                logger.info(f"Extracting knowledge from {pdf_path}")
                knowledge = await self.extractor.extract_knowledge(extracted_paper)
                
                # Save knowledge
                knowledge_path = os.path.join(self.knowledge_dir, f"{paper_id}_knowledge.json")
                self.extractor.save_knowledge(knowledge, knowledge_path)
                logger.info(f"Saved knowledge to {knowledge_path}")
                
                return knowledge
            else:
                # Return the extraction result if LLM not available
                return {
                    "paper_id": paper_id,
                    "title": extracted_paper.title,
                    "authors": extracted_paper.authors,
                    "abstract": extracted_paper.abstract,
                    "extraction_path": extraction_path,
                    "pdf_path": pdf_path,
                    "message": "LLM provider not available for knowledge extraction"
                }
                
        except Exception as e:
            logger.error(f"Error extracting from URL {url}: {str(e)}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return {"error": str(e)}
            
    async def extract_from_pdf(self, pdf_path: str, paper_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract knowledge from a local PDF file.
        
        Args:
            pdf_path (str): Path to the PDF file
            paper_id (Optional[str], optional): Custom paper ID. Defaults to None.
            
        Returns:
            Dict[str, Any]: Extracted knowledge
        """
        try:
            # Validate PDF path
            if not os.path.exists(pdf_path):
                logger.error(f"PDF file not found: {pdf_path}")
                return {"error": f"PDF file not found: {pdf_path}"}
                
            # Generate paper ID if not provided
            if not paper_id:
                paper_id = os.path.basename(pdf_path).replace('.pdf', '')
            
            # Process PDF
            logger.info(f"Processing PDF {pdf_path}")
            extracted_paper = self.processor.process_pdf(pdf_path, self.figure_dir)
            
            # Save extraction results
            extraction_path = os.path.join(self.extraction_dir, f"{paper_id}_extracted.json")
            self.processor.save_extracted_paper(extracted_paper, extraction_path)
            logger.info(f"Saved extracted paper to {extraction_path}")
            
            # Cache the extracted paper
            self.paper_cache[paper_id] = extracted_paper
            
            # Extract knowledge (if LLM provider available)
            if self.llm_provider:
                logger.info(f"Extracting knowledge from {pdf_path}")
                knowledge = await self.extractor.extract_knowledge(extracted_paper)
                
                # Save knowledge
                knowledge_path = os.path.join(self.knowledge_dir, f"{paper_id}_knowledge.json")
                self.extractor.save_knowledge(knowledge, knowledge_path)
                logger.info(f"Saved knowledge to {knowledge_path}")
                
                return knowledge
            else:
                # Return the extraction result if LLM not available
                return {
                    "paper_id": paper_id,
                    "title": extracted_paper.title,
                    "authors": extracted_paper.authors,
                    "abstract": extracted_paper.abstract,
                    "extraction_path": extraction_path,
                    "pdf_path": pdf_path,
                    "message": "LLM provider not available for knowledge extraction"
                }
                
        except Exception as e:
            logger.error(f"Error extracting from PDF {pdf_path}: {str(e)}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return {"error": str(e)}
            
    async def extract_from_urls(self, urls: List[str], max_concurrent: int = 3) -> List[Dict[str, Any]]:
        """
        Extract knowledge from multiple paper URLs concurrently.
        
        Args:
            urls (List[str]): List of URLs
            max_concurrent (int, optional): Maximum concurrent extractions. Defaults to 3.
            
        Returns:
            List[Dict[str, Any]]: List of extraction results
        """
        # Create semaphore to limit concurrent processing
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def extract_with_semaphore(url):
            async with semaphore:
                return await self.extract_from_url(url)
        
        # Create tasks for all extractions
        tasks = [extract_with_semaphore(url) for url in urls]
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        extraction_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error extracting from URL {urls[i]}: {str(result)}")
                extraction_results.append({"error": str(result), "url": urls[i]})
            else:
                extraction_results.append(result)
        
        return extraction_results
    
    async def extract_from_search_results(self, search_results: List[Dict[str, Any]], max_concurrent: int = 3) -> List[Dict[str, Any]]:
        """
        Extract knowledge from search results.
        
        Args:
            search_results (List[Dict[str, Any]]): List of search results (must have 'url' key)
            max_concurrent (int, optional): Maximum concurrent extractions. Defaults to 3.
            
        Returns:
            List[Dict[str, Any]]: List of extraction results
        """
        # Filter for results with URLs
        results_with_urls = [r for r in search_results if 'url' in r]
        
        # Extract URLs
        urls = [r['url'] for r in results_with_urls]
        
        # Extract knowledge from URLs
        results = await self.extract_from_urls(urls, max_concurrent)
        
        # Combine with original search results
        combined_results = []
        for i, result in enumerate(results):
            combined = {**results_with_urls[i], **result}
            combined_results.append(combined)
        
        return combined_results
    
    def get_paper(self, paper_id: str) -> Optional[ExtractedPaper]:
        """
        Get an extracted paper from the cache or load from file.
        
        Args:
            paper_id (str): The paper ID
            
        Returns:
            Optional[ExtractedPaper]: The extracted paper or None if not found
        """
        # Check cache first
        if paper_id in self.paper_cache:
            return self.paper_cache[paper_id]
            
        # Check if extraction file exists
        extraction_path = os.path.join(self.extraction_dir, f"{paper_id}_extracted.json")
        if os.path.exists(extraction_path):
            # Load from file
            paper = self.processor.load_extracted_paper(extraction_path)
            # Add to cache
            self.paper_cache[paper_id] = paper
            return paper
            
        return None
    
    def get_knowledge(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """
        Get extracted knowledge for a paper.
        
        Args:
            paper_id (str): The paper ID
            
        Returns:
            Optional[Dict[str, Any]]: The extracted knowledge or None if not found
        """
        # Check if knowledge file exists
        knowledge_path = os.path.join(self.knowledge_dir, f"{paper_id}_knowledge.json")
        if os.path.exists(knowledge_path):
            return self.extractor.load_knowledge(knowledge_path)
            
        return None
    
    def list_papers(self) -> List[Dict[str, Any]]:
        """
        List all processed papers with basic metadata.
        
        Returns:
            List[Dict[str, Any]]: List of papers with basic metadata
        """
        papers = []
        
        # List all extraction files
        extraction_files = [f for f in os.listdir(self.extraction_dir) if f.endswith('_extracted.json')]
        
        for file in extraction_files:
            try:
                # Extract paper ID
                paper_id = file.replace('_extracted.json', '')
                
                # Get paper
                paper = self.get_paper(paper_id)
                if paper:
                    # Check for knowledge
                    knowledge_path = os.path.join(self.knowledge_dir, f"{paper_id}_knowledge.json")
                    has_knowledge = os.path.exists(knowledge_path)
                    
                    # Add to list
                    papers.append({
                        'id': paper_id,
                        'title': paper.title,
                        'authors': paper.authors,
                        'year': paper.metadata.get('year', ''),
                        'pdf_path': paper.pdf_path,
                        'extraction_path': os.path.join(self.extraction_dir, file),
                        'has_knowledge': has_knowledge,
                        'num_sections': len(paper.sections),
                        'num_figures': len(paper.figures),
                        'num_tables': len(paper.tables),
                        'num_citations': len(paper.citations)
                    })
            except Exception as e:
                logger.error(f"Error processing paper file {file}: {str(e)}")
        
        # Sort by modification time (newest first)
        papers.sort(key=lambda p: os.path.getmtime(p['extraction_path']), reverse=True)
        
        return papers
    
    async def close(self):
        """Close connections and resources."""
        await self.retriever.close()
        
    def __del__(self):
        """Cleanup when object is deleted."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.close())
            else:
                loop.run_until_complete(self.close())
        except Exception:
            pass