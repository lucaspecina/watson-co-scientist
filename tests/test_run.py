"""
Comprehensive system test to verify all components work together.
"""

import asyncio
import logging
import shutil
import os
from pathlib import Path

from src.core.system import CoScientistSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("test_run")

async def run_system_test():
    """
    Run a comprehensive test of the entire system.
    """
    # Create a clean test directory
    test_dir = "tests/data/test_run_temp"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    Path(test_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize system
    system = CoScientistSystem(config_name="default", data_dir=test_dir)
    
    # Set a research goal focused on a complex scientific question
    research_goal = await system.analyze_research_goal(
        """
        Investigate the potential mechanisms by which microplastics may impact human health at the cellular level.
        Focus on how different types, sizes, and concentrations of microplastics interact with cellular processes, 
        immune responses, and potential long-term effects. Develop testable hypotheses that can be verified through 
        rigorous experimental approaches.
        """
    )
    
    logger.info(f"Created research goal: {research_goal.id}")
    
    # Run multiple iterations to generate hypotheses, reviews, and a research overview
    logger.info("Running system iterations...")
    
    # Run 3 iterations - this will generate hypotheses, reviews, debates, and evolution
    for i in range(3):
        logger.info(f"Starting iteration {i+1}")
        state = await system.run_iteration()
        
        # Log state information
        logger.info(f"Completed iteration {i+1}")
        logger.info(f"  Hypotheses: {state['num_hypotheses']}")
        logger.info(f"  Reviews: {state['num_reviews']}")
        logger.info(f"  Tournament matches: {state['num_tournament_matches']}")
        logger.info(f"  Protocols: {state['num_protocols']}")
    
    # Get the final research overview
    logger.info("Generating final research overview...")
    research_overview = await system._generate_research_overview()
    
    # Report on the research overview
    logger.info(f"Research overview title: {research_overview.title}")
    logger.info(f"Research areas: {len(research_overview.research_areas)}")
    
    for i, area in enumerate(research_overview.research_areas, 1):
        logger.info(f"Research area {i}: {area.get('name', 'Unnamed')}")
        
        # Check for enhanced fields
        if "causal_structure" in area:
            logger.info(f"  Has causal structure: Yes")
            
        if "verification_approach" in area:
            logger.info(f"  Has verification approach: Yes")
            
        if "testable_predictions" in area:
            predictions = area.get("testable_predictions", [])
            logger.info(f"  Testable predictions: {len(predictions)}")
    
    # Check methodological recommendations
    if research_overview.metadata and "methodological_recommendations" in research_overview.metadata:
        recommendations = research_overview.metadata["methodological_recommendations"]
        logger.info(f"Methodological recommendations: {len(recommendations)}")
    
    # Return the research overview for inspection
    return research_overview

if __name__ == "__main__":
    research_overview = asyncio.run(run_system_test())
    print("\nTest completed successfully!")
    print(f"Final research overview: {research_overview.title}")
    print(f"Number of research areas: {len(research_overview.research_areas)}")