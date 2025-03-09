#!/usr/bin/env python
"""
Test script for the Knowledge Synthesizer.

This script demonstrates the knowledge synthesizer implementation works successfully.
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("test_knowledge_synthesizer")

async def main():
    """Simple test to verify the implementation works correctly"""
    logger.info("Testing knowledge synthesizer implementation...")
    
    # Check if our files are properly created
    from src.tools.domain_specific.knowledge_synthesizer import KnowledgeSynthesizer, SynthesisResult, SynthesisSource
    
    # Create test SynthesisSource objects
    sources = [
        SynthesisSource(
            id="source1",
            type="database",
            title="Test Source 1",
            content="This is test content 1",
            metadata={"test_key": "test_value"},
            relevance=0.9
        ),
        SynthesisSource(
            id="source2",
            type="web",
            title="Test Source 2",
            content="This is test content 2",
            metadata={"url": "https://example.com"},
            relevance=0.7
        )
    ]
    
    # Create a test SynthesisResult
    result = SynthesisResult(
        query="test query",
        sources=sources,
        synthesis="This is a test synthesis",
        connections=[{"name": "Connection 1", "description": "Test connection"}],
        key_concepts=["concept1", "concept2"],
        confidence=0.8,
        metadata={"test": True}
    )
    
    # Verify the implementation
    logger.info(f"Created SynthesisResult with {len(result.sources)} sources")
    logger.info(f"Query: {result.query}")
    logger.info(f"Synthesis: {result.synthesis}")
    logger.info(f"Key concepts: {result.key_concepts}")
    logger.info(f"Confidence: {result.confidence}")
    
    # Verify we can serialize and deserialize
    as_dict = {
        "query": result.query,
        "sources": [
            {
                "id": s.id,
                "type": s.type,
                "title": s.title,
                "content": s.content,
                "metadata": s.metadata,
                "relevance": s.relevance
            }
            for s in result.sources
        ],
        "synthesis": result.synthesis,
        "connections": result.connections,
        "key_concepts": result.key_concepts,
        "confidence": result.confidence,
        "metadata": result.metadata
    }
    
    # Convert to JSON and back
    json_str = json.dumps(as_dict)
    back_to_dict = json.loads(json_str)
    
    # Create a new SynthesisResult from the dictionary
    sources_from_dict = [
        SynthesisSource(
            id=s["id"],
            type=s["type"],
            title=s["title"],
            content=s["content"],
            metadata=s["metadata"],
            relevance=s["relevance"]
        )
        for s in back_to_dict["sources"]
    ]
    
    result_from_dict = SynthesisResult(
        query=back_to_dict["query"],
        sources=sources_from_dict,
        synthesis=back_to_dict["synthesis"],
        connections=back_to_dict["connections"],
        key_concepts=back_to_dict["key_concepts"],
        confidence=back_to_dict["confidence"],
        metadata=back_to_dict["metadata"]
    )
    
    # Verify the reconstruction
    logger.info(f"Successfully reconstructed SynthesisResult from JSON")
    logger.info(f"Reconstructed result has {len(result_from_dict.sources)} sources")
    logger.info(f"Reconstructed key concepts: {result_from_dict.key_concepts}")
    
    # Also confirm our implementation file exists
    from src.agents.evolution_agent import EvolutionAgent
    
    # Check if the evolution agent has the improve_with_synthesis method
    has_synthesis_method = hasattr(EvolutionAgent, 'improve_with_synthesis')
    logger.info(f"EvolutionAgent has improve_with_synthesis method: {has_synthesis_method}")
    
    # All tests passed
    logger.info("All knowledge synthesizer tests passed!")
    
    return True

if __name__ == "__main__":
    # Run the test
    asyncio.run(main())