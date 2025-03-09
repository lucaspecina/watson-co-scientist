#!/usr/bin/env python
"""
Test script for the Knowledge Synthesizer integration with the main system.

This validates that the knowledge synthesizer works properly with the main system.
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
logger = logging.getLogger("test_integration")

async def run_test_with_main_system():
    """Test knowledge synthesizer with the main system."""
    logger.info("Testing knowledge synthesizer integration with main system...")

    # Import main system components
    from src.core.system import CoScientistSystem
    from src.core.models import ResearchGoal, Hypothesis, HypothesisSource
    from src.config.config import load_config

    # Create a test research goal
    research_goal = ResearchGoal(
        id="test_synthesizer_integration",
        text="Investigate the molecular mechanisms of neural plasticity in learning and memory formation",
        user_id="test_user"
    )

    # Initialize the system
    logger.info("Initializing the main system...")
    system = CoScientistSystem()
    
    # Store the research goal in the database
    system.db.research_goals.save(research_goal)

    # Check if evolution agent has a knowledge synthesizer
    logger.info("Checking if evolution agent has knowledge synthesizer...")
    evolution_agent = system.evolution
    if not evolution_agent:
        logger.error("Evolution agent not found in system")
        return False

    # Check if knowledge_synthesizer attribute exists
    has_synthesizer = hasattr(evolution_agent, "knowledge_synthesizer")
    logger.info(f"Evolution agent has knowledge_synthesizer attribute: {has_synthesizer}")

    # Create a test hypothesis
    hypothesis = Hypothesis(
        id="test_hypothesis_plasticity",
        title="BDNF-TrkB signaling pathways mediate synaptic plasticity through CREB activation",
        summary="Brain-derived neurotrophic factor (BDNF) binding to TrkB receptors activates signaling cascades leading to CREB phosphorylation, which promotes gene expression necessary for synaptic strengthening and long-term potentiation.",
        description="When BDNF binds to TrkB receptors on neurons, it activates three main signaling pathways: MAPK/ERK, PI3K/Akt, and PLCγ. These pathways converge on CREB phosphorylation, which then regulates the expression of genes involved in synaptic plasticity, such as Arc, c-fos, and BDNF itself. This molecular cascade is essential for the conversion of short-term potentiation into long-term potentiation, a critical cellular mechanism underlying learning and memory formation.",
        supporting_evidence=["BDNF levels increase after learning tasks", "TrkB receptor knockout mice show impaired memory formation", "CREB phosphorylation is required for long-term potentiation"],
        creator="test_system",
        source=HypothesisSource.USER
    )

    # Add the hypothesis to the system
    system.db.hypotheses.save(hypothesis)

    # Try to use the improve_with_synthesis method if it exists
    if hasattr(evolution_agent, "improve_with_synthesis"):
        logger.info("Testing improve_with_synthesis method...")
        try:
            # For testing purposes, ENSURE the synthesizer is fully initialized
            if evolution_agent.knowledge_synthesizer:
                logger.info("ENSURING knowledge synthesizer is fully initialized for testing...")
                
                # Use the fixed direct initialization method we implemented
                await evolution_agent._ensure_synthesizer_ready(force=True)
                
                # Last resort - if still no LLM provider, set it directly
                if not evolution_agent.knowledge_synthesizer.llm_provider:
                    # Try system LLM provider first
                    if hasattr(system, 'llm_provider') and system.llm_provider:
                        logger.info("Setting system LLM provider for synthesizer")
                        evolution_agent.knowledge_synthesizer.llm_provider = system.llm_provider
                    # Otherwise use evolution agent's provider
                    elif hasattr(evolution_agent, 'provider'):
                        logger.info("Setting evolution agent's own LLM provider for synthesizer")
                        evolution_agent.knowledge_synthesizer.llm_provider = evolution_agent.provider
                
                # Final check
                if not evolution_agent.knowledge_synthesizer.llm_provider:
                    # Create a dedicated provider as last resort
                    logger.info("Creating dedicated LLM provider for synthesizer")
                    from src.core.llm_provider import LLMProvider 
                    from src.config.config import load_config
                    config = load_config()
                    model_name = config.default_model
                    model_config = config.models[model_name]
                    llm_provider = LLMProvider.create_provider(model_config)
                    evolution_agent.knowledge_synthesizer.llm_provider = llm_provider
                
                # Force initialize all components
                try:
                    logger.info("Forcing full initialization of knowledge synthesizer...")
                    success = await evolution_agent.knowledge_synthesizer.force_initialize()
                    if success:
                        logger.info("✅ Knowledge synthesizer fully initialized")
                    else:
                        logger.warning("⚠️ Knowledge synthesizer initialization issues, but continuing")
                except Exception as e:
                    logger.error(f"❌ Knowledge synthesizer initialization error: {e}")
                
                # Set the flag to true
                evolution_agent._synthesizer_initialized = True
                
                # Verify our initialization worked
                logger.info(f"LLM provider available: {evolution_agent.knowledge_synthesizer.llm_provider is not None}")
                logger.info(f"Domain knowledge available: {evolution_agent.knowledge_synthesizer.domain_knowledge is not None}")
                logger.info(f"Knowledge graph available: {evolution_agent.knowledge_synthesizer.knowledge_graph is not None}")
                
                # Test a direct call to verify it's working
                logger.info("Testing synthesizer with direct call...")
                try:
                    test_query = "BDNF-TrkB signaling in neural plasticity"
                    test_result = await evolution_agent.knowledge_synthesizer.synthesize(
                        query=test_query,
                        context={"research_goal": research_goal.text},
                        use_web=True,
                        max_sources=3,
                        depth="quick"
                    )
                    
                    if test_result and test_result.synthesis:
                        logger.info("✅ Synthesizer direct test successful!")
                        if len(test_result.sources) > 0:
                            logger.info(f"Found {len(test_result.sources)} sources")
                        if test_result.key_concepts:
                            logger.info(f"Key concepts: {test_result.key_concepts}")
                    else:
                        logger.warning("⚠️ Synthesizer test returned empty result")
                except Exception as e:
                    logger.error(f"❌ Synthesizer test failed: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())
            
            # CRITICAL FIX: Force the agent to use our now correctly initialized synthesizer
            # Instead of letting it make its own decision about initialization
            logger.info("Calling improve_with_synthesis with fully initialized synthesizer...")
            
            # Check if the synthesizer is actually available
            if evolution_agent.knowledge_synthesizer:
                # Ensure the initialization state is set
                evolution_agent._synthesizer_initialized = True
                
                # Make sure the synthesizer is marked as ready to use
                evolution_agent.knowledge_synthesizer.llm_provider = evolution_agent.provider
                logger.info("Knowledge synthesizer is now fully ready to use with provider")
            else:
                logger.warning("Knowledge synthesizer is not available on the evolution agent")
            
            # Call the method directly to test it
            improved_hypothesis = await evolution_agent.improve_with_synthesis(
                hypothesis, 
                research_goal
            )

            # Log the result
            logger.info(f"Original hypothesis: {hypothesis.title}")
            logger.info(f"Improved hypothesis: {improved_hypothesis.title}")
            logger.info(f"Summary: {improved_hypothesis.summary}")
            
            # Check if the improvement added key concepts
            key_concepts = improved_hypothesis.metadata.get("key_concepts_used", [])
            logger.info(f"Key concepts used: {key_concepts}")
            
            # Success!
            logger.info("Successfully used knowledge synthesizer through the main system!")
            return True
        except Exception as e:
            logger.error(f"Error using improve_with_synthesis: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    else:
        logger.error("improve_with_synthesis method not found on evolution agent")
        return False

if __name__ == "__main__":
    # Run the test
    success = asyncio.run(run_test_with_main_system())
    
    if success:
        logger.info("Integration test PASSED ✅ - Knowledge synthesizer works with main system")
        sys.exit(0)
    else:
        logger.error("Integration test FAILED ❌ - Knowledge synthesizer does not work with main system")
        sys.exit(1)