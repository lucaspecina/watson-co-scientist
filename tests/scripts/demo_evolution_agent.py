#!/usr/bin/env python
"""
Demonstration script for the Evolution Agent using the Knowledge Graph.

This script demonstrates how the Evolution Agent can use the knowledge graph
to improve hypotheses by analyzing papers, extracting knowledge, and using
that knowledge to enhance scientific hypotheses.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# Ensure parent directory is in the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import components
from src.agents.evolution_agent import EvolutionAgent
from src.core.models import Hypothesis, ResearchGoal, Review, HypothesisSource
from src.core.llm_provider import LLMProvider
from src.config.config import load_config, SystemConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("demo_evolution_agent")

async def demo_evolution_agent():
    """
    Demonstrate the Evolution Agent's ability to improve hypotheses using the knowledge graph.
    """
    # Load config
    config = load_config()
    
    # Initialize LLM provider
    model_config = config.models[config.default_model]
    llm_provider = LLMProvider.create_provider(model_config)
    
    # Create system config (use the loaded config directly)
    system_config = config
    
    # Add paper extraction specific configuration
    setattr(system_config, "paper_extraction_enabled", True)
    setattr(system_config, "paper_extraction_dir", "data/papers_db")
    setattr(system_config, "knowledge_graph_dir", "data/knowledge_graph")
    
    # Initialize evolution agent
    evolution_agent = EvolutionAgent(system_config)
    
    # Set the LLM provider explicitly
    evolution_agent.llm_provider = llm_provider
    
    # Create a test hypothesis about antimicrobial resistance and horizontal gene transfer
    hypothesis = Hypothesis(
        id="hyp1",
        title="Horizontal Gene Transfer via Plasmids as Primary AMR Spread Mechanism",
        summary="Conjugative plasmids serve as the predominant vectors for horizontal gene transfer of antimicrobial resistance genes in gram-negative bacteria.",
        description="""
        This hypothesis proposes that conjugative plasmids are the primary mechanism for horizontal gene transfer (HGT) of antimicrobial resistance (AMR) genes in gram-negative bacteria. 
        These self-transmissible plasmids efficiently transfer between bacterial cells through direct contact, enabling the rapid spread of resistance genes across bacterial populations. 
        The process is facilitated by pili formation and DNA replication systems encoded by the plasmids themselves. 
        Due to their ability to transfer independently of chromosomal replication and across species barriers, conjugative plasmids represent a more significant threat for AMR dissemination compared to other HGT mechanisms such as transformation and transduction.
        """,
        supporting_evidence=[
            "Multiple AMR genes are frequently co-located on single conjugative plasmids",
            "Conjugative plasmids can transfer between diverse bacterial species",
            "Plasmid-mediated resistance has been documented for most major antibiotic classes"
        ],
        creator="user",
        source=HypothesisSource.USER
    )
    
    # Create a research goal
    research_goal = ResearchGoal(
        id="goal1",
        text="Investigate the mechanisms of horizontal gene transfer in gram-negative bacteria and their contribution to antimicrobial resistance spread."
    )
    
    # Print initial information
    logger.info("Starting demonstration of Evolution Agent with Knowledge Graph")
    logger.info(f"Initial hypothesis: {hypothesis.title}")
    logger.info(f"Research goal: {research_goal.text}")
    
    # Checking if paper extraction and knowledge graph are initialized
    logger.info("Checking if paper extraction system is initialized...")
    if hasattr(evolution_agent, 'extraction_manager') and evolution_agent.extraction_manager:
        logger.info("Paper extraction manager is initialized.")
    else:
        logger.error("Paper extraction manager is not initialized!")
    
    if hasattr(evolution_agent, 'knowledge_graph') and evolution_agent.knowledge_graph:
        logger.info(f"Knowledge graph is initialized with {len(evolution_agent.knowledge_graph.entities)} entities, " 
                   f"{len(evolution_agent.knowledge_graph.relations)} relations, and "
                   f"{len(evolution_agent.knowledge_graph.papers)} papers.")
    else:
        logger.error("Knowledge graph is not initialized!")
    
    # Now improve the hypothesis using the knowledge graph
    logger.info("Improving hypothesis using the knowledge graph...")
    improved_hypothesis = await evolution_agent.improve_with_knowledge_graph(
        hypothesis=hypothesis,
        research_goal=research_goal,
        reviews=None  # No reviews for this demo
    )
    
    # Print the improved hypothesis
    logger.info("\n=== IMPROVED HYPOTHESIS ===")
    logger.info(f"Title: {improved_hypothesis.title}")
    logger.info(f"Summary: {improved_hypothesis.summary}")
    logger.info("Description:")
    logger.info(improved_hypothesis.description)
    logger.info("Supporting evidence:")
    for evidence in improved_hypothesis.supporting_evidence:
        logger.info(f"- {evidence}")
    
    # Print metadata about which entities and insights were used
    if improved_hypothesis.metadata:
        logger.info("\nEntities used from knowledge graph:")
        for entity in improved_hypothesis.metadata.get("entities_used", []):
            logger.info(f"- {entity}")
            
        logger.info("\nInsights applied from knowledge graph:")
        for insight in improved_hypothesis.metadata.get("insights_applied", []):
            logger.info(f"- {insight}")
            
        logger.info("\nNovel connections:")
        for connection in improved_hypothesis.metadata.get("novel_connections", []):
            logger.info(f"- {connection}")
    
    # Display graph statistics after improvement
    if hasattr(evolution_agent, 'knowledge_graph') and evolution_agent.knowledge_graph:
        logger.info(f"\nFinal knowledge graph statistics: {len(evolution_agent.knowledge_graph.entities)} entities, " 
                   f"{len(evolution_agent.knowledge_graph.relations)} relations, and "
                   f"{len(evolution_agent.knowledge_graph.papers)} papers.")
    
    return True

if __name__ == "__main__":
    asyncio.run(demo_evolution_agent())