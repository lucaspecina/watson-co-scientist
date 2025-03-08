#!/usr/bin/env python3
"""
Test script for multi-domain scientific database integration.
This script tests the new cross-domain functionality in the Watson Co-Scientist system.
"""

import os
import asyncio
import json
from src.tools.domain_specific.knowledge_manager import DomainKnowledgeManager
from src.tools.domain_specific.cross_domain_synthesizer import CrossDomainSynthesizer

async def test_domain_detection():
    """Test domain detection from research questions."""
    print("\n===== Testing Domain Detection =====")
    
    # Create the cross-domain synthesizer
    manager = DomainKnowledgeManager()
    synthesizer = CrossDomainSynthesizer(manager)
    
    # Test with different research questions
    test_queries = [
        "The role of mitochondria in Alzheimer's disease",
        "Using machine learning for drug discovery",
        "Quantum algorithms for protein folding simulations",
        "Chemical synthesis of novel antibiotics"
    ]
    
    for query in test_queries:
        domains = synthesizer.detect_research_domains(query)
        print(f"\nQuery: {query}")
        print(f"Detected domains: {domains}")

async def test_domain_search():
    """Test searching across multiple scientific domains."""
    print("\n===== Testing Domain Search =====")
    
    # Create the knowledge manager
    manager = DomainKnowledgeManager()
    
    # Initialize manager for necessary domains
    domains = ["biomedicine", "computer_science", "chemistry"]
    print(f"Initializing providers for domains: {domains}")
    await manager.initialize(domains=domains)
    
    # Test queries
    queries = [
        "machine learning alzheimer's",
        "mitochondrial dysfunction parkinson's"
    ]
    
    for query in queries:
        print(f"\nSearching for: {query}")
        results = await manager.search(query, domains=domains, limit=2)
        
        for domain, domain_results in results.items():
            print(f"\n{domain.upper()} ({len(domain_results)} results):")
            
            for i, result in enumerate(domain_results, 1):
                print(f"  {i}. {result.get('title', 'No title')}")
                if result.get("authors"):
                    print(f"     Authors: {', '.join(result['authors'][:2])}")
                print(f"     Source: {result.get('provider', 'unknown')}")

async def test_cross_domain_synthesis():
    """Test cross-domain knowledge synthesis."""
    print("\n===== Testing Cross-Domain Synthesis =====")
    
    # Create the knowledge manager and synthesizer
    manager = DomainKnowledgeManager()
    synthesizer = CrossDomainSynthesizer(manager)
    
    # Initialize domains
    domains = ["biomedicine", "biology", "computer_science"]
    print(f"Initializing providers for domains: {domains}")
    await manager.initialize(domains=domains)
    
    # Test synthesis query
    query = "machine learning applications in neurodegenerative disease treatment"
    print(f"\nSynthesizing knowledge for: {query}")
    
    # Get domain detection
    domains = synthesizer.detect_research_domains(query)
    print(f"Detected domains: {domains}")
    
    # Do a quick search to demonstrate cross-domain information retrieval
    results = await manager.search(query, domains=[d for d, _ in domains if d in ["biomedicine", "computer_science"]], limit=2)
    
    for domain, domain_results in results.items():
        print(f"\n{domain.upper()} ({len(domain_results)} results):")
        for i, result in enumerate(domain_results, 1):
            print(f"  {i}. {result.get('title', 'No title')}")
            print(f"     Source: {result.get('provider', 'unknown')}")

async def main():
    """Run all tests."""
    print("===== Testing Multi-Domain Scientific Database Integration =====")
    
    # Test domain detection
    await test_domain_detection()
    
    # Test domain search
    try:
        await test_domain_search()
    except Exception as e:
        print(f"Error in domain search: {e}")
    
    # Test cross-domain synthesis
    try:
        await test_cross_domain_synthesis()
    except Exception as e:
        print(f"Error in cross-domain synthesis: {e}")
    
    print("\n===== All Tests Completed =====")

if __name__ == "__main__":
    asyncio.run(main())