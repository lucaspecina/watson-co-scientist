#!/usr/bin/env python3
"""
Test script to run the Watson Co-Scientist system with a specified research goal.
"""

import asyncio
import sys
import os
from dotenv import load_dotenv

# Add src to the path so we can import from it
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# Import the system
from src.core.system import CoScientistSystem
from src.utils.logger import setup_logger

# Load environment variables
load_dotenv()

async def main():
    """Run a test of the system with a specified research goal."""
    # Set up logging
    logger = setup_logger()
    logger.info("Starting Watson Co-Scientist test run")
    
    # Initialize the system
    system = CoScientistSystem()
    
    # Set the research goal
    research_goal = "Develop new treatment approaches for Alzheimer's disease focusing on amyloid-beta clearance mechanisms"
    logger.info(f"Setting research goal: {research_goal}")
    
    # Analyze the research goal
    await system.analyze_research_goal(research_goal)
    
    # Run a few iterations
    num_iterations = 2
    logger.info(f"Running {num_iterations} iterations")
    
    for i in range(num_iterations):
        logger.info(f"Starting iteration {i+1}/{num_iterations}")
        await system.run_iteration()
        logger.info(f"Completed iteration {i+1}/{num_iterations}")
        
        # Print the current state
        logger.info(f"Current state: {system.current_state}")
        
    logger.info("Test run completed successfully")
    
    # Print summary information
    print("\n==== Test Run Summary ====")
    print(f"Research Goal: {research_goal}")
    print(f"Iterations completed: {system.current_state['iterations_completed']}")
    print(f"Hypotheses generated: {system.current_state['num_hypotheses']}")
    print(f"Reviews completed: {system.current_state['num_reviews']}")
    print(f"Tournament matches: {system.current_state['num_tournament_matches']}")
    print(f"Experimental protocols: {system.current_state['num_protocols']}")
    
    # Get top hypotheses
    if system.current_state['top_hypotheses']:
        print("\nTop Hypotheses:")
        for i, h_id in enumerate(system.current_state['top_hypotheses'][:3], 1):
            hypothesis = system.db.hypotheses.get(h_id)
            if hypothesis:
                print(f"  {i}. {hypothesis.title}")
    
    # Get experimental protocols
    protocols = system.db.experimental_protocols.get_all()
    if protocols:
        print("\nExperimental Protocols:")
        for i, protocol in enumerate(protocols[:3], 1):
            # Get the hypothesis for this protocol
            hypothesis = system.db.hypotheses.get(protocol.hypothesis_id)
            hypothesis_title = hypothesis.title if hypothesis else "Unknown"
            
            print(f"  {i}. {protocol.title}")
            print(f"     For hypothesis: {hypothesis_title}")
            print(f"     Steps: {len(protocol.steps)}")
    
if __name__ == "__main__":
    asyncio.run(main())