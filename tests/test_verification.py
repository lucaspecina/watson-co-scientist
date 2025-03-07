"""
Test script for the enhanced deep verification and simulation capabilities.
"""

import os
import asyncio
import logging
from datetime import datetime

from src.core.system import CoScientistSystem
from src.core.models import (
    ResearchGoal, 
    Hypothesis, 
    Review,
    HypothesisStatus,
    HypothesisSource
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("test_verification")

async def test_run():
    """Run a simple test of the enhanced verification features."""
    
    # Initialize system
    system = CoScientistSystem(config_name="default", data_dir="data_test")
    
    # Define a test research goal focusing on complex hypothesis verification
    research_goal_text = """
    Investigate the potential role of mitochondrial dysfunction in the pathogenesis of neurodegenerative diseases, 
    with a focus on reactive oxygen species (ROS) production, calcium homeostasis disruption, and their impact 
    on neuronal cell death pathways. Develop testable hypotheses about the causal relationships between 
    mitochondrial dysfunction and disease progression that can be verified through rigorous experimental approaches.
    """
    
    # Create research goal
    research_goal = await system.analyze_research_goal(research_goal_text)
    logger.info(f"Created research goal: {research_goal.id}")
    
    # Create a high-quality test hypothesis
    test_hypothesis = Hypothesis(
        title="Mitochondrial Complex I Dysfunction Initiates a Feed-Forward Cycle of Calcium Dysregulation and ROS Production in Neurodegeneration",
        description="""
        This hypothesis proposes that initial defects in mitochondrial Complex I activity trigger a self-amplifying cycle of 
        reactive oxygen species (ROS) production and calcium dysregulation that drives neurodegeneration. 
        
        The proposed mechanism works as follows:
        
        1. Initial Complex I dysfunction (due to genetic, environmental, or age-related factors) leads to electron leakage 
        and increased superoxide production at the inner mitochondrial membrane.
        
        2. This initial ROS production oxidizes nearby calcium channels in the mitochondrial membranes (particularly MCU - 
        mitochondrial calcium uniporter, and VDAC - voltage-dependent anion channel), altering their function and increasing 
        calcium influx into the mitochondria.
        
        3. Elevated mitochondrial calcium levels further inhibit Complex I activity through direct binding and conformational 
        changes to Complex I subunits, particularly ND5 and NDUFS7.
        
        4. This calcium-induced Complex I inhibition leads to more ROS production, creating a feed-forward cycle.
        
        5. As this cycle progresses, excessive calcium accumulation in mitochondria triggers opening of the mitochondrial 
        permeability transition pore (mPTP), leading to mitochondrial swelling, cytochrome c release, and initiation of 
        apoptotic cell death pathways.
        
        6. Neurons with high energy demands and extended axonal mitochondrial networks (such as motor neurons affected in ALS, 
        dopaminergic neurons in Parkinson's, or cortical neurons in Alzheimer's) are particularly vulnerable to this cascade.
        
        This mechanism explains the selective vulnerability of certain neuronal populations, the progressive nature of 
        neurodegeneration, and the apparent convergence of multiple neurodegenerative disorders on mitochondrial dysfunction.
        """,
        summary="Mitochondrial Complex I dysfunction initiates a self-amplifying cycle where ROS production leads to calcium channel oxidation, increasing mitochondrial calcium influx, which further inhibits Complex I, producing more ROS. This feed-forward cycle ultimately triggers mPTP opening, cytochrome c release, and neuronal apoptosis, explaining the progressive nature of neurodegenerative diseases.",
        supporting_evidence=[
            "Complex I deficiency has been documented in post-mortem brain tissue from patients with Parkinson's disease, Alzheimer's disease, and ALS",
            "Calcium dysregulation is a common feature across neurodegenerative disorders",
            "ROS scavengers and calcium chelators have shown protective effects in animal models of neurodegeneration",
            "Mutations in genes encoding Complex I subunits increase risk for several neurodegenerative diseases"
        ],
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
    
    # Extract probability assessment
    if deep_verification_review.metadata and "probability_correct" in deep_verification_review.metadata:
        logger.info(f"Probability assessment: {deep_verification_review.metadata['probability_correct']}%")
    
    # Extract verification experiments
    if deep_verification_review.metadata and "verification_experiments" in deep_verification_review.metadata:
        experiments = deep_verification_review.metadata["verification_experiments"]
        logger.info(f"Suggested verification experiments: {len(experiments)}")
        for i, exp in enumerate(experiments[:3], 1):
            logger.info(f"  Experiment {i}: {exp}")
    
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
    
    # Test verification analysis in meta-review
    verification_analysis = await system.meta_review.analyze_verification_reviews(
        [deep_verification_review],
        [simulation_review],
        research_goal
    )
    
    logger.info("Verification analysis results:")
    logger.info(f"  Causal reasoning patterns: {len(verification_analysis.get('causal_reasoning_patterns', []))}")
    logger.info(f"  Probability insights: {len(verification_analysis.get('probability_insights', []))}")
    logger.info(f"  Verification experiments: {len(verification_analysis.get('verification_experiments', []))}")
    logger.info(f"  Simulation insights: {len(verification_analysis.get('simulation_insights', []))}")
    
    # Generate a research overview with verification insights
    overview = await system.meta_review.generate_research_overview(
        research_goal,
        [test_hypothesis],
        None,
        None,
        None,
        verification_analysis
    )
    
    logger.info(f"Generated research overview: {overview.id}")
    logger.info(f"Title: {overview.title}")
    logger.info(f"Research areas: {len(overview.research_areas)}")
    
    # Check for enhanced fields in research areas
    for i, area in enumerate(overview.research_areas, 1):
        logger.info(f"Research area {i}: {area.get('name', 'Unnamed')}")
        
        # Check for enhanced verification fields
        if "causal_structure" in area:
            logger.info(f"  Has causal structure description: Yes")
        if "verification_approach" in area:
            logger.info(f"  Has verification approach: Yes")
        if "testable_predictions" in area and area.get("testable_predictions"):
            logger.info(f"  Testable predictions: {len(area.get('testable_predictions'))}")
        if "potential_failure_modes" in area and area.get("potential_failure_modes"):
            logger.info(f"  Potential failure modes: {len(area.get('potential_failure_modes'))}")
    
    # Check for methodological recommendations
    if overview.metadata and "methodological_recommendations" in overview.metadata:
        recommendations = overview.metadata["methodological_recommendations"]
        logger.info(f"Methodological recommendations: {len(recommendations)}")
        for i, rec in enumerate(recommendations[:3], 1):
            logger.info(f"  Recommendation {i}: {rec}")
    
    logger.info("Test completed successfully")
    return overview

if __name__ == "__main__":
    asyncio.run(test_run())