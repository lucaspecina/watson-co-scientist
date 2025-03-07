"""
Simplified test script for the enhanced deep verification capabilities.
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

logger = logging.getLogger("test_verification_simple")

async def test_run():
    """Run a simple test of the deep verification feature."""
    
    # Initialize system
    system = CoScientistSystem(config_name="default", data_dir="data_test")
    
    # Define a test research goal
    research_goal_text = "Investigate the potential role of mitochondrial dysfunction in neurodegenerative diseases."
    
    # Create research goal
    research_goal = await system.analyze_research_goal(research_goal_text)
    logger.info(f"Created research goal: {research_goal.id}")
    
    # Create a test hypothesis
    test_hypothesis = Hypothesis(
        title="Mitochondrial Complex I Dysfunction and Neurodegeneration",
        description="Mitochondrial Complex I dysfunction leads to increased ROS production.",
        summary="Mitochondrial Complex I dysfunction leads to neurodegeneration via ROS production.",
        supporting_evidence=["Complex I deficiency found in neurodegenerative diseases"],
        creator="generation",
        status=HypothesisStatus.GENERATED,
        source=HypothesisSource.SYSTEM,
        citations=[]
    )
    
    # Save to database
    system.db.hypotheses.save(test_hypothesis)
    logger.info(f"Created test hypothesis: {test_hypothesis.id}")
    
    # Test deep verification review
    deep_verification_review = await system.reflection.deep_verification_review(
        test_hypothesis,
        research_goal
    )
    
    # Save the review
    system.db.reviews.save(deep_verification_review)
    logger.info(f"Completed deep verification review: {deep_verification_review.id}")
    logger.info(f"Deep verification score: {deep_verification_review.overall_score}")
    
    logger.info("Test completed successfully")
    return deep_verification_review

if __name__ == "__main__":
    asyncio.run(test_run())