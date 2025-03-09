#!/usr/bin/env python3
"""
Watson Co-Scientist: An AI system to assist scientists in hypothesis generation and research planning.
This is the main entry point for the application.
"""

import argparse
import os
import sys
from dotenv import load_dotenv

# Add src to the path so we can import from it
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# Import our modules
from src.core.system import CoScientistSystem
from src.utils.logger import setup_logger

# Load environment variables
load_dotenv()

async def main_async():
    """Async main entry point for the Co-Scientist system."""
    # Set up logging
    logger = setup_logger()
    logger.info("Starting Watson Co-Scientist system")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Watson Co-Scientist: AI system to assist in scientific research")
    parser.add_argument("--research_goal", type=str, help="The research goal to analyze")
    parser.add_argument("--config", type=str, default="default", help="Configuration to use")
    parser.add_argument("--run", type=int, help="Number of iterations to run")
    args = parser.parse_args()
    
    try:
        # Initialize the system
        system = CoScientistSystem(config_name=args.config)
        
        # If a research goal was provided, analyze it
        if args.research_goal:
            await system.analyze_research_goal(args.research_goal)
            
            # If run iterations were specified, run them
            if args.run and args.run > 0:
                logger.info(f"Running {args.run} iteration(s)...")
                for i in range(args.run):
                    logger.info(f"Starting iteration {i+1}/{args.run}")
                    state = await system.run_iteration()
                    logger.info(f"Completed iteration {i+1}/{args.run}")
                    logger.info(f"  Hypotheses: {state['num_hypotheses']}")
                    logger.info(f"  Reviews: {state['num_reviews']}")
                    logger.info(f"  Tournament matches: {state['num_tournament_matches']}")
                    
                # Generate a research overview after all iterations
                logger.info("Generating research overview...")
                overview = await system._generate_research_overview()
                if overview:
                    logger.info(f"Research overview generated: {overview.title}")
                    logger.info(f"Research areas: {len(overview.research_areas)}")
                else:
                    logger.info("No research overview generated")
        else:
            # Otherwise start interactive mode
            await system.start_interactive_mode()
            
    except Exception as e:
        logger.error(f"Error running Co-Scientist system: {e}", exc_info=True)
        return 1
        
    return 0

def main():
    """Main entry point for the Co-Scientist system."""
    import asyncio
    
    # Run the async main function
    return asyncio.run(main_async())

if __name__ == "__main__":
    sys.exit(main())