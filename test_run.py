#!/usr/bin/env python3
"""
Test script to run the Co-Scientist system for a few iterations.
"""

import asyncio
import sys
import os
from dotenv import load_dotenv

# Add src to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# Import modules
from src.core.system import CoScientistSystem
from src.utils.logger import setup_logger

# Load environment variables
load_dotenv()

async def main():
    """Run the system for a few iterations."""
    # Set up logging
    logger = setup_logger()
    logger.info("Starting test run")
    
    # Initialize the system
    system = CoScientistSystem()
    
    # Set research goal
    research_goal = await system.analyze_research_goal(
        "Investigate the role of mitochondrial dysfunction in neurodegenerative diseases"
    )
    
    # Run a few iterations
    for i in range(3):
        print(f"Running iteration {i+1}/3")
        await system.run_iteration()
        
    # Generate a research overview
    print("Generating research overview")
    overview = await system._generate_research_overview()
    
    if overview:
        print(f"Generated research overview: {overview.title}")
    else:
        print("Failed to generate research overview")
    
    print("Done")

if __name__ == "__main__":
    asyncio.run(main())