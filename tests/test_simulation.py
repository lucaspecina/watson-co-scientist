"""
Test script for the enhanced simulation capabilities.
"""

import asyncio
import logging
from datetime import datetime

from src.core.system import CoScientistSystem
from src.core.models import (
    ResearchGoal, 
    Hypothesis, 
    HypothesisStatus,
    HypothesisSource
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("test_simulation")

async def test_run():
    """Run a simple test of the simulation feature."""
    
    # Initialize system
    system = CoScientistSystem(config_name="default", data_dir="tests/data/small_dataset")
    
    # Define a test research goal
    research_goal_text = "Investigate the potential role of mitochondrial dysfunction in neurodegenerative diseases."
    
    # Create research goal
    research_goal = await system.analyze_research_goal(research_goal_text)
    logger.info(f"Created research goal: {research_goal.id}")
    
    # Create a test hypothesis
    test_hypothesis = Hypothesis(
        title="Mitochondrial Complex I Dysfunction and Neurodegeneration",
        description="""
        This hypothesis proposes that initial defects in mitochondrial Complex I activity trigger a series of downstream effects:
        
        1. Complex I dysfunction leads to increased ROS production
        2. ROS damages mitochondrial DNA and proteins
        3. This leads to mitochondrial permeability transition pore opening
        4. Cytochrome c is released, activating caspase pathways
        5. Neuronal apoptosis occurs, causing neurodegenerative symptoms
        """,
        summary="Mitochondrial Complex I dysfunction leads to neurodegeneration via ROS production and apoptotic pathways.",
        supporting_evidence=["Complex I deficiency found in neurodegenerative diseases"],
        creator="generation",
        status=HypothesisStatus.GENERATED,
        source=HypothesisSource.SYSTEM,
        citations=[]
    )
    
    # Save to database
    system.db.hypotheses.save(test_hypothesis)
    logger.info(f"Created test hypothesis: {test_hypothesis.id}")
    
    # Test simulation review
    simulation_review = await system.reflection.simulation_review(
        test_hypothesis,
        research_goal
    )
    
    # Save the review
    system.db.reviews.save(simulation_review)
    logger.info(f"Completed simulation review: {simulation_review.id}")
    logger.info(f"Simulation score: {simulation_review.overall_score}")
    
    # Extract simulation insights
    if simulation_review.metadata:
        if "model_description" in simulation_review.metadata:
            logger.info(f"Model description: {simulation_review.metadata['model_description'][:100]}...")
            
        if "predictions" in simulation_review.metadata:
            predictions = simulation_review.metadata["predictions"]
            logger.info(f"Model predictions: {len(predictions)}")
            for i, pred in enumerate(predictions[:3], 1):
                logger.info(f"  Prediction {i}: {pred.get('prediction', 'No prediction text')}")
        
        if "emergent_properties" in simulation_review.metadata:
            props = simulation_review.metadata["emergent_properties"]
            logger.info(f"Emergent properties: {len(props)}")
            for i, prop in enumerate(props[:3], 1):
                logger.info(f"  Property {i}: {prop}")
                
        if "confidence_score" in simulation_review.metadata:
            logger.info(f"Confidence score: {simulation_review.metadata['confidence_score']}%")
    
    logger.info("Test completed successfully")
    return simulation_review

if __name__ == "__main__":
    asyncio.run(test_run())