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
from core.system import CoScientistSystem
from utils.logger import setup_logger

# Load environment variables
load_dotenv()

def main():
    """Main entry point for the Co-Scientist system."""
    # Set up logging
    logger = setup_logger()
    logger.info("Starting Watson Co-Scientist system")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Watson Co-Scientist: AI system to assist in scientific research")
    parser.add_argument("--research_goal", type=str, help="The research goal to analyze")
    parser.add_argument("--config", type=str, default="default", help="Configuration to use")
    args = parser.parse_args()
    
    try:
        # Initialize the system
        system = CoScientistSystem(config_name=args.config)
        
        # If a research goal was provided, analyze it
        if args.research_goal:
            system.analyze_research_goal(args.research_goal)
        else:
            # Otherwise start interactive mode
            system.start_interactive_mode()
            
    except Exception as e:
        logger.error(f"Error running Co-Scientist system: {e}", exc_info=True)
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())