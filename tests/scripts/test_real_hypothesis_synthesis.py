#\!/usr/bin/env python
"""
Test a real scientific hypothesis improvement using the knowledge synthesizer.

This script directly tests the improve_with_synthesis method of the EvolutionAgent
by creating the necessary objects and calling the method directly.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
import json
import uuid

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("sci_test")

async def run_real_scientific_test():
    """Test knowledge synthesizer with a real scientific hypothesis."""
    # Import necessary components
    from src.core.llm_provider import LLMProvider
    from src.config.config import load_config
    from src.agents.evolution_agent import EvolutionAgent
    from src.tools.domain_specific.knowledge_synthesizer import KnowledgeSynthesizer
    from src.core.models import ResearchGoal, Hypothesis, HypothesisSource
    
    # Generate a unique test ID
    test_id = str(uuid.uuid4())[:8]
    logger.info(f"Starting real scientific test (ID: {test_id})...")
    
    # Load configuration
    config = load_config()
    
    # Create the evolution agent
    evolution_agent = EvolutionAgent(config)
    
    # Ensure we have a provider
    if not hasattr(evolution_agent, "provider"):
        logger.error("Evolution agent has no LLM provider")
        return False
    
    # Create a synthesizer directly
    synthesizer_config = {
        "storage_dir": "data/syntheses",
        "knowledge_graph": {"storage_dir": "data/knowledge_graph"},
        "domain_knowledge": config.get("domain_knowledge", {})
    }
    
    # Create directories
    for dir_path in [synthesizer_config["storage_dir"], synthesizer_config["knowledge_graph"]["storage_dir"]]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Create synthesizer
    synthesizer = KnowledgeSynthesizer(evolution_agent.provider, synthesizer_config)
    
    # Initialize the synthesizer
    logger.info("Initializing synthesizer...")
    await synthesizer.force_initialize()
    
    # Set it on the evolution agent
    evolution_agent.knowledge_synthesizer = synthesizer
    evolution_agent._synthesizer_initialized = True
    
    # Create a research goal
    research_goal = ResearchGoal(
        id=f"test_synthesizer_{test_id}",
        text="Explore the role of Histone Deacetylases (HDACs) in regulating gene expression during long-term memory formation",
        user_id="test_user"
    )
    
    # Create a hypothesis
    hypothesis = Hypothesis(
        id=f"test_hypothesis_{test_id}",
        title="HDAC inhibition enhances memory formation through histone acetylation",
        summary="Inhibition of Histone Deacetylases (HDACs) leads to increased histone acetylation, promoting gene expression necessary for long-term memory consolidation.",
        description="Histone deacetylases (HDACs) are enzymes that remove acetyl groups from histone proteins, generally leading to gene silencing. HDAC inhibitors prevent this deacetylation, resulting in a more open chromatin structure and increased gene expression. This hypothesis proposes that HDAC inhibition specifically enhances memory formation by allowing the expression of memory-related genes during the consolidation phase.",
        supporting_evidence=["HDAC inhibitors enhance long-term potentiation in hippocampal slices", "HDAC2 knockout mice show enhanced memory performance", "Memory-related genes show increased expression after HDAC inhibition"],
        creator="test_system",
        source=HypothesisSource.USER
    )
    
    # Log the test plan
    logger.info(f"Testing improve_with_synthesis on hypothesis: {hypothesis.title}")
    
    # Run the improvement
    try:
        logger.info("Calling improve_with_synthesis...")
        improved_hypothesis = await evolution_agent.improve_with_synthesis(
            hypothesis, 
            research_goal
        )
        
        # Log the result
        logger.info(f"Original hypothesis: {hypothesis.title}")
        logger.info(f"Improved hypothesis: {improved_hypothesis.title}")
        logger.info(f"Summary: {improved_hypothesis.summary}")
        
        # Check if we got key concepts
        key_concepts = improved_hypothesis.metadata.get("key_concepts_used", [])
        logger.info(f"Key concepts used: {key_concepts}")
        synthesis_insights = improved_hypothesis.metadata.get("synthesis_insights_applied", [])
        logger.info(f"Synthesis insights: {synthesis_insights}")
        
        # Success - real synthesis used?
        if key_concepts or synthesis_insights:
            logger.info("✅ Successfully used real synthesis in the improvement\!")
            return True
        else:
            logger.warning("⚠️ Improved hypothesis doesn't contain synthesis data")
            return False
    except Exception as e:
        logger.error(f"❌ Error during hypothesis improvement: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    # Run the test
    success = asyncio.run(run_real_scientific_test())
    
    if success:
        logger.info("Real scientific test PASSED ✅ - Knowledge synthesizer works\!")
        sys.exit(0)
    else:
        logger.error("Real scientific test FAILED ❌ - Knowledge synthesizer doesn't work properly")
        sys.exit(1)
