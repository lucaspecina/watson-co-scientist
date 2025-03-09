"""
Test script for the meta-review verification analysis.
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

logger = logging.getLogger("test_verification_analysis")

async def test_run():
    """Run a test of the verification analysis capability."""
    
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
    
    # Generate both types of reviews
    deep_verification_review = await system.reflection.deep_verification_review(
        test_hypothesis,
        research_goal
    )
    system.db.reviews.save(deep_verification_review)
    logger.info(f"Generated deep verification review: {deep_verification_review.id}")
    
    simulation_review = await system.reflection.simulation_review(
        test_hypothesis,
        research_goal
    )
    system.db.reviews.save(simulation_review)
    logger.info(f"Generated simulation review: {simulation_review.id}")
    
    # Run verification analysis
    verification_analysis = await system.meta_review.analyze_verification_reviews(
        [deep_verification_review],
        [simulation_review],
        research_goal
    )
    
    # Log verification analysis results
    logger.info("Verification analysis results:")
    logger.info(f"Causal reasoning patterns: {len(verification_analysis.get('causal_reasoning_patterns', []))}")
    
    for i, pattern in enumerate(verification_analysis.get('causal_reasoning_patterns', [])[:3], 1):
        logger.info(f"  Pattern {i}: {pattern.get('pattern', 'Unknown')} (Impact: {pattern.get('impact', 'Unknown')})")
    
    logger.info(f"Probability insights: {len(verification_analysis.get('probability_insights', []))}")
    for i, insight in enumerate(verification_analysis.get('probability_insights', [])[:3], 1):
        logger.info(f"  Insight {i}: {insight}")
    
    logger.info(f"Verification experiments: {len(verification_analysis.get('verification_experiments', []))}")
    for i, exp in enumerate(verification_analysis.get('verification_experiments', [])[:3], 1):
        logger.info(f"  Experiment {i}: {exp.get('experiment', 'Unknown')}")
    
    logger.info(f"Simulation insights: {len(verification_analysis.get('simulation_insights', []))}")
    for i, insight in enumerate(verification_analysis.get('simulation_insights', [])[:3], 1):
        logger.info(f"  Insight {i}: {insight.get('insight', 'Unknown')}")
    
    logger.info(f"Common failure modes: {len(verification_analysis.get('common_failure_modes', []))}")
    for i, mode in enumerate(verification_analysis.get('common_failure_modes', [])[:3], 1):
        logger.info(f"  Mode {i}: {mode.get('failure_mode', 'Unknown')}")
    
    # Generate research overview with verification insights
    overview = await system.meta_review.generate_research_overview(
        research_goal,
        [test_hypothesis],
        None, 
        None,
        None,
        verification_analysis
    )
    
    # Log overview details
    logger.info(f"Generated research overview: {overview.id}")
    logger.info(f"Overview title: {overview.title}")
    logger.info(f"Research areas: {len(overview.research_areas)}")
    
    # Check enhanced fields in research areas
    for i, area in enumerate(overview.research_areas, 1):
        logger.info(f"Research area {i}: {area.get('name', 'Unnamed')}")
        
        if "causal_structure" in area:
            logger.info(f"  Has causal structure: Yes")
            causal_structure = area.get("causal_structure", "")
            logger.info(f"  Causal structure: {causal_structure[:100]}...")
            
        if "verification_approach" in area:
            logger.info(f"  Has verification approach: Yes")
            
        if "testable_predictions" in area:
            predictions = area.get("testable_predictions", [])
            logger.info(f"  Testable predictions: {len(predictions)}")
            
        if "potential_failure_modes" in area:
            failure_modes = area.get("potential_failure_modes", [])
            logger.info(f"  Potential failure modes: {len(failure_modes)}")
    
    # Check methodological recommendations
    if overview.metadata and "methodological_recommendations" in overview.metadata:
        recommendations = overview.metadata["methodological_recommendations"]
        logger.info(f"Methodological recommendations: {len(recommendations)}")
        for i, rec in enumerate(recommendations[:3], 1):
            logger.info(f"  Recommendation {i}: {rec}")
    
    logger.info("Test completed successfully")
    return verification_analysis, overview

if __name__ == "__main__":
    asyncio.run(test_run())