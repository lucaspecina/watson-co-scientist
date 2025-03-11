#\!/usr/bin/env python
"""
Test script for the Knowledge Synthesizer with direct LLM provider usage.
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("direct_test")

# Add project root to path
project_root = str(Path(__file__).resolve().parent.parent.parent)
print(f"Adding project root to path: {project_root}")
sys.path.insert(0, project_root)

async def main():
    """Run the direct test for knowledge synthesizer."""
    from src.tools.domain_specific.knowledge_synthesizer import KnowledgeSynthesizer
    from src.core.llm_provider import LLMProvider
    from src.config.config import load_config, ModelConfig
    
    logger.info("Starting direct test of knowledge synthesizer")
    
    # Load config
    config = load_config()
    
    # Get LLM model config
    model_name = config.default_model
    model_config = config.models[model_name]
    
    # Create LLM provider
    logger.info(f"Creating LLM provider with model {model_name}")
    llm_provider = LLMProvider.create_provider(model_config)
    
    # Create knowledge synthesizer
    synthesizer_config = {
        "storage_dir": "data/syntheses",
        "knowledge_graph": {"storage_dir": "data/knowledge_graph"},
        "domain_knowledge": config.get("domain_knowledge", {})
    }
    
    # Ensure directories exist
    os.makedirs(synthesizer_config["storage_dir"], exist_ok=True)
    os.makedirs(synthesizer_config["knowledge_graph"]["storage_dir"], exist_ok=True)
    
    # Create synthesizer
    logger.info("Creating knowledge synthesizer")
    synthesizer = KnowledgeSynthesizer(llm_provider, synthesizer_config)
    
    # Initialize
    success = await synthesizer.force_initialize()
    logger.info(f"Initialization: {success}")
    
    # Test synthesize method
    logger.info("Testing synthesis - wait for results...")
    result = await synthesizer.synthesize(
        query="BDNF-TrkB signaling in synaptic plasticity and memory formation",
        use_web=True, 
        depth="quick",
        max_sources=5
    )
    
    # Show results
    logger.info(f"Synthesis successful: {result is not None}")
    if result:
        logger.info(f"Sources: {len(result.sources)}")
        logger.info(f"Key concepts: {result.key_concepts}")
        logger.info(f"Synthesis snippet: {result.synthesis[:200]}...")
    
    return True

if __name__ == "__main__":
    asyncio.run(main())

