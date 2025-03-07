#!/usr/bin/env python3
"""
Test script to run the Co-Scientist system with user interaction capabilities.
"""

import asyncio
import sys
import os
import uuid
from dotenv import load_dotenv

# Add src to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# Import modules
from src.core.system import CoScientistSystem
from src.utils.logger import setup_logger
from src.core.models import (
    UserFeedback, 
    Hypothesis, 
    HypothesisSource,
    ResearchFocus,
    Review,
    ReviewType
)

# Load environment variables
load_dotenv()

async def main():
    """Run the system with scientist-in-the-loop interactions."""
    # Set up logging
    logger = setup_logger()
    logger.info("Starting test run")
    
    # Initialize the system
    system = CoScientistSystem()
    
    # Create a test user ID
    user_id = str(uuid.uuid4())
    print(f"Test user ID: {user_id}")
    
    # Set research goal (this would normally come from a scientist)
    research_goal = await system.analyze_research_goal(
        "Investigate the role of mitochondrial dysfunction in neurodegenerative diseases"
    )
    
    # Run first iteration to generate initial hypotheses
    print("Running iteration 1/4")
    await system.run_iteration()
    
    # Simulate a scientist submitting their own hypothesis
    print("\nSubmitting user hypothesis...")
    user_hypothesis = Hypothesis(
        title="Mitochondrial Calcium Dysregulation as a Driver of Neuronal Death in Neurodegenerative Disorders",
        description="""This hypothesis proposes that disruption in mitochondrial calcium handling plays a central role in neurodegeneration. Mitochondria function as intracellular calcium buffers, and dysfunction in calcium sequestration can lead to cytosolic calcium overload, triggering excitotoxicity and apoptotic cascades. The sustained calcium dysregulation may further impair mitochondrial function by disrupting the membrane potential, increasing ROS production, and compromising ATP synthesis. This creates a vicious cycle of mitochondrial damage and calcium dysregulation that ultimately leads to neuronal death characteristic of neurodegenerative diseases.""",
        summary="Disruption in mitochondrial calcium handling mechanisms contributes to neuronal death in neurodegenerative diseases by triggering excitotoxicity and apoptotic pathways.",
        supporting_evidence=[
            "Elevated cytosolic calcium levels have been observed in multiple neurodegenerative disease models",
            "Mitochondrial calcium uniporter dysfunction has been linked to neurodegeneration",
            "Calcium dysregulation correlates with increased ROS production in degenerating neurons"
        ],
        creator="user",
        source=HypothesisSource.USER,
        user_id=user_id,
        metadata={"research_goal_id": research_goal.id}
    )
    system.db.hypotheses.save(user_hypothesis)
    print(f"  Saved user hypothesis: {user_hypothesis.id}")
    
    # Add researcher feedback to guide the process
    print("Adding user feedback...")
    user_feedback = UserFeedback(
        research_goal_id=research_goal.id,
        feedback_type="direction",
        text="I'd like to see more focus on the connection between mitochondrial dysfunction and neuroinflammation, particularly the role of microglia and astrocytes in mediating inflammatory responses triggered by mitochondrial damage.",
        user_id=user_id
    )
    system.db.user_feedback.save(user_feedback)
    print(f"  Saved user feedback: {user_feedback.id}")
    
    # Add a specific research focus area
    print("Adding research focus area...")
    focus_area = ResearchFocus(
        research_goal_id=research_goal.id,
        title="Mitochondrial Dysfunction in Neuroinflammation",
        description="Investigate how mitochondrial damage and dysfunction in neurons and glial cells contribute to neuroinflammatory processes in neurodegenerative diseases.",
        keywords=["neuroinflammation", "microglia", "astrocytes", "NLRP3", "inflammasome"],
        priority=0.8,
        user_id=user_id,
        active=True
    )
    system.db.research_focus.save(focus_area)
    print(f"  Saved research focus: {focus_area.id}")
    
    # Run next iteration - the system should now prioritize reviewing the user hypothesis
    # and consider the research focus and feedback
    print("\nRunning iteration 2/4")
    await system.run_iteration()
    
    # User reviews a hypothesis
    print("\nSubmitting user review...")
    # Find a hypothesis to review
    all_hypotheses = system.db.hypotheses.get_all()
    if all_hypotheses:
        system_hypothesis = next((h for h in all_hypotheses if h.source == HypothesisSource.SYSTEM), None)
        if system_hypothesis:
            user_review = Review(
                hypothesis_id=system_hypothesis.id,
                review_type=ReviewType.USER,
                reviewer="user",
                text=f"This hypothesis has merit, but it could be improved by considering the role of mitochondrial fission/fusion dynamics in the process. The relationship between mitochondrial morphology and function is critical to understanding neurodegenerative mechanisms.",
                novelty_score=7.5,
                correctness_score=8.0,
                testability_score=9.0,
                overall_score=8.0,
                critiques=["Does not address mitochondrial dynamics sufficiently", 
                         "Needs more connection to human patient data"],
                strengths=["Well-grounded in current literature", 
                         "Provides a clear mechanistic pathway"],
                improvement_suggestions=["Include discussion of mitochondrial fission/fusion proteins", 
                                       "Connect to clinical observations"],
                user_id=user_id
            )
            system.db.reviews.save(user_review)
            print(f"  Saved user review for hypothesis: {system_hypothesis.id}")
    
    # Run next iterations
    print("\nRunning iteration 3/4")
    await system.run_iteration()
    
    print("\nRunning iteration 4/4")
    await system.run_iteration()
    
    # Generate a research overview
    print("\nGenerating research overview...")
    overview = await system._generate_research_overview()
    
    if overview:
        print(f"Generated research overview: {overview.title}")
        
        # Count user hypotheses in top results
        user_hypotheses_in_top = sum(1 for h_id in overview.top_hypotheses 
                                   if system.db.hypotheses.get(h_id) and 
                                   system.db.hypotheses.get(h_id).source == HypothesisSource.USER)
        print(f"User hypotheses in top results: {user_hypotheses_in_top}/{len(overview.top_hypotheses)}")
        
        # Check if research focus was addressed
        focus_keywords_found = False
        for area in overview.research_areas:
            area_text = area.get('description', '') + ' ' + area.get('name', '')
            if any(keyword.lower() in area_text.lower() for keyword in focus_area.keywords):
                focus_keywords_found = True
                break
                
        print(f"Research focus addressed in overview: {focus_keywords_found}")
    else:
        print("Failed to generate research overview")
    
    print("\nScientist-in-the-loop demonstration completed successfully")

if __name__ == "__main__":
    asyncio.run(main())