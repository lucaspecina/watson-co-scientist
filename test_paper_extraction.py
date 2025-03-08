#!/usr/bin/env python3
"""
Test script for the Paper Knowledge Extraction System.

This script demonstrates the capabilities of the Paper Knowledge Extraction System
by retrieving, processing, and extracting knowledge from scientific papers.
"""

import os
import asyncio
import argparse
import json
from pprint import pprint
from datetime import datetime
from dotenv import load_dotenv

# Add src to the path so we can import from it
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# Import necessary components
from src.core.llm_provider import get_llm_provider
from src.tools.paper_extraction.extraction_manager import PaperExtractionManager
from src.utils.logger import setup_logger

# Load .env variables
load_dotenv()

async def extract_from_url(url, paper_id=None, llm_provider=None):
    """Extract knowledge from a paper URL."""
    print(f"\n===== Extracting knowledge from URL: {url} =====")
    
    # Configure extraction manager
    config = {
        'base_dir': 'data/papers_db'
    }
    
    # Create extraction manager
    manager = PaperExtractionManager(config, llm_provider=llm_provider)
    
    # Extract from URL
    result = await manager.extract_from_url(url, paper_id)
    
    # Clean up
    await manager.close()
    
    return result

async def extract_from_pdf(pdf_path, paper_id=None, llm_provider=None):
    """Extract knowledge from a local PDF file."""
    print(f"\n===== Extracting knowledge from PDF: {pdf_path} =====")
    
    # Configure extraction manager
    config = {
        'base_dir': 'data/papers_db'
    }
    
    # Create extraction manager
    manager = PaperExtractionManager(config, llm_provider=llm_provider)
    
    # Extract from PDF
    result = await manager.extract_from_pdf(pdf_path, paper_id)
    
    # Clean up
    await manager.close()
    
    return result

async def list_papers():
    """List all processed papers."""
    print("\n===== Listing processed papers =====")
    
    # Configure extraction manager
    config = {
        'base_dir': 'data/papers_db'
    }
    
    # Create extraction manager
    manager = PaperExtractionManager(config)
    
    # List papers
    papers = manager.list_papers()
    
    # Display papers
    print(f"Found {len(papers)} processed papers:")
    for i, paper in enumerate(papers, 1):
        print(f"\n{i}. {paper['title']}")
        print(f"   Authors: {', '.join(paper['authors'])}")
        print(f"   Year: {paper['year']}")
        print(f"   Sections: {paper['num_sections']}")
        print(f"   Figures: {paper['num_figures']}")
        print(f"   Tables: {paper['num_tables']}")
        print(f"   Citations: {paper['num_citations']}")
        print(f"   Has knowledge: {paper['has_knowledge']}")
    
    # Clean up
    await manager.close()
    
    return papers

async def main():
    """Main function."""
    # Set up logging
    logger = setup_logger()
    
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Test Paper Knowledge Extraction System')
    parser.add_argument('--url', help='URL to a scientific paper')
    parser.add_argument('--pdf', help='Path to a local PDF file')
    parser.add_argument('--id', help='Custom paper ID')
    parser.add_argument('--list', action='store_true', help='List all processed papers')
    parser.add_argument('--no-llm', action='store_true', help='Disable LLM-based extraction')
    
    args = parser.parse_args()
    
    # Initialize LLM provider if requested
    llm_provider = None
    if not args.no_llm:
        try:
            llm_provider = get_llm_provider()
            print("LLM provider initialized successfully")
        except Exception as e:
            print(f"Failed to initialize LLM provider: {e}")
            print("Running without LLM-based extraction")
    
    # Determine the action to take
    if args.list:
        # List all processed papers
        await list_papers()
    elif args.url:
        # Extract from URL
        result = await extract_from_url(args.url, args.id, llm_provider)
        
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print("\nExtraction completed successfully!")
            print(f"Title: {result.get('title', 'Unknown')}")
            print(f"Authors: {', '.join(result.get('authors', []))}")
            
            # Print some highlights
            if "entities" in result and result["entities"]:
                print("\nTop entities:")
                for i, entity in enumerate(sorted(result["entities"], 
                                                key=lambda x: x.get("importance_score", 0), 
                                                reverse=True)[:5], 1):
                    print(f"  {i}. {entity.get('name', 'Unknown')} ({entity.get('type', 'Unknown')})")
                    print(f"     {entity.get('definition', 'No definition')}")
            
            if "findings" in result and result["findings"]:
                print("\nKey findings:")
                for i, finding in enumerate(result["findings"][:3], 1):
                    print(f"  {i}. {finding.get('finding', 'Unknown')}")
                    
            if "methods" in result and result["methods"]:
                print("\nKey methods:")
                for i, method in enumerate(result["methods"][:3], 1):
                    print(f"  {i}. {method.get('method', 'Unknown')}")
    
    elif args.pdf:
        # Extract from PDF
        result = await extract_from_pdf(args.pdf, args.id, llm_provider)
        
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print("\nExtraction completed successfully!")
            print(f"Title: {result.get('title', 'Unknown')}")
            print(f"Authors: {', '.join(result.get('authors', []))}")
            
            # Print some highlights
            if "entities" in result and result["entities"]:
                print("\nTop entities:")
                for i, entity in enumerate(sorted(result["entities"], 
                                                key=lambda x: x.get("importance_score", 0), 
                                                reverse=True)[:5], 1):
                    print(f"  {i}. {entity.get('name', 'Unknown')} ({entity.get('type', 'Unknown')})")
                    print(f"     {entity.get('definition', 'No definition')}")
            
            if "findings" in result and result["findings"]:
                print("\nKey findings:")
                for i, finding in enumerate(result["findings"][:3], 1):
                    print(f"  {i}. {finding.get('finding', 'Unknown')}")
                    
            if "methods" in result and result["methods"]:
                print("\nKey methods:")
                for i, method in enumerate(result["methods"][:3], 1):
                    print(f"  {i}. {method.get('method', 'Unknown')}")
    else:
        # No arguments - show usage
        parser.print_help()

if __name__ == "__main__":
    asyncio.run(main())