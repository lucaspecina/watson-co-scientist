#!/usr/bin/env python3
"""
Test script specifically for the ArXiv provider.
This script tests the ArXiv API integration in detail.
"""

import os
import sys
import json
import asyncio
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_arxiv")

# Import the ArXiv provider
from src.tools.domain_specific.providers.arxiv_provider import ArxivProvider

async def test_arxiv_initialization():
    """Test ArXiv provider initialization."""
    print("\n===== Testing ArXiv Provider Initialization =====")
    
    # Create provider
    provider = ArxivProvider()
    
    # Test initialization
    success = await provider.initialize()
    
    if success:
        print("✅ ArXiv provider initialized successfully")
    else:
        print("❌ ArXiv provider initialization failed")
        print(f"Debug info:")
        print(f"Provider is initialized: {provider._is_initialized}")

async def test_arxiv_search():
    """Test searching with the ArXiv provider."""
    print("\n===== Testing ArXiv Search =====")
    
    # Create provider
    provider = ArxivProvider()
    
    # Initialize provider
    initialized = await provider.initialize()
    if not initialized:
        print("❌ Skipping search test as provider could not be initialized")
        return
    
    # Test with a well-formed query that should return results
    queries = [
        "quantum computing",  # Computer science
        "higgs boson",        # Physics
        "deep learning"       # CS/ML
    ]
    
    for query in queries:
        print(f"\nSearching for: '{query}'")
        results = await provider.search(query, limit=3)
        
        if results:
            print(f"✅ Found {len(results)} results")
            for i, result in enumerate(results, 1):
                print(f"  {i}. {result.get('title', 'Unknown')}")
                if result.get('authors'):
                    print(f"     Authors: {', '.join(result['authors'][:2])}")
                if result.get('url'):
                    print(f"     URL: {result['url']}")
        else:
            print(f"❌ No results found for '{query}'")

async def test_arxiv_detailed_queries():
    """Test more detailed ArXiv search queries."""
    print("\n===== Testing ArXiv Detailed Queries =====")
    
    # Create provider
    provider = ArxivProvider()
    
    # Initialize provider
    initialized = await provider.initialize()
    if not initialized:
        print("❌ Skipping detailed query test as provider could not be initialized")
        return
    
    # Test with more complex queries
    detailed_queries = [
        # Category-specific query
        "cat:cs.AI machine learning",
        # Author-specific query
        "au:\"Hinton, Geoffrey\"",
        # Title-specific query
        "ti:\"neural networks\""
    ]
    
    for query in detailed_queries:
        print(f"\nSearching with detailed query: '{query}'")
        results = await provider.search(query, limit=3)
        
        if results:
            print(f"✅ Found {len(results)} results")
            for i, result in enumerate(results, 1):
                print(f"  {i}. {result.get('title', 'Unknown')}")
                if result.get('authors'):
                    print(f"     Authors: {', '.join(result['authors'][:2])}")
                if result.get('categories'):
                    print(f"     Categories: {', '.join(result['categories'][:3])}")
        else:
            print(f"❌ No results found for '{query}'")

async def test_arxiv_entity_fetching():
    """Test fetching specific ArXiv papers by ID."""
    print("\n===== Testing ArXiv Entity Fetching =====")
    
    # Create provider
    provider = ArxivProvider()
    
    # Initialize provider
    initialized = await provider.initialize()
    if not initialized:
        print("❌ Skipping entity fetching test as provider could not be initialized")
        return
    
    # Well-known ArXiv IDs to test
    paper_ids = [
        "2303.08774",  # GPT-4 paper
        "1706.03762"   # Transformer paper
    ]
    
    for paper_id in paper_ids:
        print(f"\nFetching paper with ID: {paper_id}")
        paper = await provider.get_entity(paper_id)
        
        if paper:
            print(f"✅ Successfully fetched paper: {paper.get('title', 'Unknown')}")
            if paper.get('authors'):
                print(f"   Authors: {', '.join(paper['authors'][:3])}")
            if paper.get('abstract'):
                abstract = paper['abstract']
                # Truncate abstract if too long
                if len(abstract) > 200:
                    abstract = abstract[:200] + "..."
                print(f"   Abstract: {abstract}")
        else:
            print(f"❌ Failed to fetch paper with ID {paper_id}")

async def test_arxiv_related_entities():
    """Test finding related papers."""
    print("\n===== Testing ArXiv Related Entities =====")
    
    # Create provider
    provider = ArxivProvider()
    
    # Initialize provider
    initialized = await provider.initialize()
    if not initialized:
        print("❌ Skipping related entities test as provider could not be initialized")
        return
    
    # Use a well-known paper ID
    paper_id = "1706.03762"  # Transformer paper
    
    print(f"\nFinding papers related to: {paper_id}")
    related = await provider.get_related_entities(paper_id, limit=3)
    
    if related:
        print(f"✅ Found {len(related)} related papers")
        for i, paper in enumerate(related, 1):
            print(f"  {i}. {paper.get('title', 'Unknown')}")
            if paper.get('authors'):
                print(f"     Authors: {', '.join(paper['authors'][:2])}")
    else:
        print(f"❌ No related papers found for {paper_id}")

async def test_arxiv_citation_formatting():
    """Test citation formatting for ArXiv papers."""
    print("\n===== Testing ArXiv Citation Formatting =====")
    
    # Create provider
    provider = ArxivProvider()
    
    # Initialize provider
    initialized = await provider.initialize()
    if not initialized:
        print("❌ Skipping citation test as provider could not be initialized")
        return
    
    # Get a paper to cite
    paper_id = "2303.08774"  # GPT-4 paper
    paper = await provider.get_entity(paper_id)
    
    if not paper:
        print(f"❌ Could not fetch paper with ID {paper_id} for citation test")
        return
    
    # Test different citation styles
    styles = ["apa", "mla", "chicago", "vancouver"]
    
    for style in styles:
        citation = provider.format_citation(paper, style=style)
        print(f"\n{style.upper()} style citation:")
        print(f"  {citation}")

async def main():
    """Run all ArXiv provider tests."""
    print("===== Starting ArXiv Provider Tests =====")
    
    # Run initialization test
    await test_arxiv_initialization()
    
    # Run search test
    await test_arxiv_search()
    
    # Run detailed query test
    await test_arxiv_detailed_queries()
    
    # Run entity fetching test
    await test_arxiv_entity_fetching()
    
    # Run related entities test
    await test_arxiv_related_entities()
    
    # Run citation formatting test
    await test_arxiv_citation_formatting()
    
    print("\n===== All ArXiv Provider Tests Completed =====")

if __name__ == "__main__":
    # Load environment variables if needed
    load_dotenv()
    
    # Run the tests
    asyncio.run(main())