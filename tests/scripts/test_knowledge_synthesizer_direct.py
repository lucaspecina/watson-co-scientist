#!/usr/bin/env python
"""
Test script that directly uses the Knowledge Synthesizer without the full system.

This creates a simple test environment to ensure the synthesizer works.
"""

import os
import sys
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
logger = logging.getLogger("test_synthesizer")

async def run_direct_test():
    """Test knowledge synthesizer directly."""
    logger.info("Testing knowledge synthesizer directly...")

    # Import necessary components
    from src.tools.domain_specific.knowledge_synthesizer import KnowledgeSynthesizer, SynthesisResult, SynthesisSource
    from src.core.llm_provider import LLMProvider
    from src.config.config import load_config
    from src.tools.domain_specific.knowledge_manager import DomainKnowledgeManager
    from src.tools.paper_extraction.knowledge_graph import KnowledgeGraph

    # Create minimal configuration
    config = load_config()
    
    # Set up LLM provider (this is the key part)
    logger.info("Setting up LLM provider...")
    llm_provider = LLMProvider(config.get('llm', {}))
    
    # Configure the synthesizer
    synthesizer_config = {
        'knowledge_graph': {'storage_dir': 'data/knowledge_graph'},
        'domain_knowledge': config.get('domain_knowledge', {}),
        'storage_dir': 'data/syntheses'
    }
    
    # Ensure directories exist
    os.makedirs(synthesizer_config['knowledge_graph']['storage_dir'], exist_ok=True)
    os.makedirs(synthesizer_config['storage_dir'], exist_ok=True)
    
    # Create and initialize the synthesizer
    logger.info("Creating and initializing knowledge synthesizer...")
    synthesizer = KnowledgeSynthesizer(llm_provider, synthesizer_config)
    
    # Initialize components
    logger.info("Initializing domain knowledge...")
    await synthesizer.domain_knowledge.initialize()
    
    # Force initialize the synthesizer
    success = await synthesizer.force_initialize()
    logger.info(f"Force initialization {'succeeded' if success else 'failed'}")
    
    # Run a test synthesis
    logger.info("Running test synthesis...")
    query = "BDNF-TrkB signaling in neural plasticity and memory formation"
    result = await synthesizer.synthesize(
        query=query,
        context={"test_purpose": "Validating synthesizer functionality"},
        use_kg=True,
        use_domains=True,
        use_web=True,  # Try with web search
        max_sources=5,
        depth="quick"  # Use quick for faster testing
    )
    
    # Log the results
    if result:
        logger.info(f"Synthesis completed successfully!")
        logger.info(f"Query: {result.query}")
        logger.info(f"Number of sources: {len(result.sources)}")
        logger.info(f"Source types: {[s.type for s in result.sources]}")
        logger.info(f"Key concepts: {result.key_concepts}")
        logger.info(f"Synthesis snippet: {result.synthesis[:200]}...")
        return True
    else:
        logger.error("Synthesis failed to produce a result")
        return False

if __name__ == "__main__":
    # Run the test
    success = asyncio.run(run_direct_test())
    
    if success:
        logger.info("Direct test PASSED ✅ - Knowledge synthesizer works!")
        sys.exit(0)
    else:
        logger.error("Direct test FAILED ❌ - Knowledge synthesizer does not work")
        sys.exit(1)