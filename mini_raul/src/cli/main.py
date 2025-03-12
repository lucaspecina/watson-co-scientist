"""
Main CLI entry point for mini-RAUL.
"""
import os
import sys
import json
import logging
import argparse
import asyncio
from typing import Dict, Any, List, Optional, Union

from ..core.session import Session
from ..core.coordinator import Coordinator
from . import commands

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


async def main_async():
    """Main async entry point."""
    parser = argparse.ArgumentParser(description='mini-RAUL: Research Co-Scientist')
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # create-session command
    create_parser = subparsers.add_parser('create-session', help='Create a new research session')
    create_parser.add_argument('session_id', help='ID for the new session')
    create_parser.add_argument('--data-dir', help='Directory to store session data')
    
    # list-sessions command
    list_parser = subparsers.add_parser('list-sessions', help='List all available sessions')
    list_parser.add_argument('--data-dir', help='Directory containing session data')
    
    # start-research command
    start_parser = subparsers.add_parser('start-research', help='Start a new research process')
    start_parser.add_argument('session_id', help='ID of the session to use')
    start_parser.add_argument('research_goal', help='The research goal')
    start_parser.add_argument('--preferences', default='', help='User preferences')
    start_parser.add_argument('--constraints', default='', help='User constraints')
    start_parser.add_argument('--data-dir', help='Directory containing session data')
    
    # continue-research command
    continue_parser = subparsers.add_parser('continue-research', help='Continue an existing research process')
    continue_parser.add_argument('session_id', help='ID of the session to use')
    continue_parser.add_argument('--feedback', default='', help='User feedback for the previous iteration')
    continue_parser.add_argument('--data-dir', help='Directory containing session data')
    
    # get-meta-review command
    meta_parser = subparsers.add_parser('get-meta-review', help='Get a meta-review of the research progress')
    meta_parser.add_argument('session_id', help='ID of the session to use')
    meta_parser.add_argument('--data-dir', help='Directory containing session data')
    
    # session-status command
    status_parser = subparsers.add_parser('session-status', help='Print the status of a session')
    status_parser.add_argument('session_id', help='ID of the session to use')
    status_parser.add_argument('--data-dir', help='Directory containing session data')
    
    # print-hypotheses command
    hypotheses_parser = subparsers.add_parser('print-hypotheses', help='Print the hypotheses for a session')
    hypotheses_parser.add_argument('session_id', help='ID of the session to use')
    hypotheses_parser.add_argument('--iteration', type=int, help='Iteration number to print hypotheses for')
    hypotheses_parser.add_argument('--data-dir', help='Directory containing session data')
    
    # print-rankings command
    rankings_parser = subparsers.add_parser('print-rankings', help='Print the rankings for a session')
    rankings_parser.add_argument('session_id', help='ID of the session to use')
    rankings_parser.add_argument('--iteration', type=int, help='Iteration number to print rankings for')
    rankings_parser.add_argument('--data-dir', help='Directory containing session data')
    
    # print-reflections command
    reflections_parser = subparsers.add_parser('print-reflections', help='Print the reflections for a session')
    reflections_parser.add_argument('session_id', help='ID of the session to use')
    reflections_parser.add_argument('--iteration', type=int, help='Iteration number to print reflections for')
    reflections_parser.add_argument('--data-dir', help='Directory containing session data')
    
    # print-evolutions command
    evolutions_parser = subparsers.add_parser('print-evolutions', help='Print the evolutions for a session')
    evolutions_parser.add_argument('session_id', help='ID of the session to use')
    evolutions_parser.add_argument('--iteration', type=int, help='Iteration number to print evolutions for')
    evolutions_parser.add_argument('--data-dir', help='Directory containing session data')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        # Execute the appropriate command
        if args.command == 'create-session':
            session = await commands.create_session(args.session_id, args.data_dir)
            print(f"Created session: {session.session_id}")
        
        elif args.command == 'list-sessions':
            sessions = await commands.list_sessions(args.data_dir)
            
            if not sessions:
                print("No sessions found")
            else:
                print(f"Found {len(sessions)} sessions:")
                for session in sessions:
                    print(f"  {session['session_id']} - {session['state']} - {session['research_goal'][:50]}...")
        
        elif args.command == 'start-research':
            session = await commands.load_session(args.session_id, args.data_dir)
            coordinator = Coordinator(session)
            await commands.start_research(coordinator, args.research_goal, args.preferences, args.constraints)
            print(f"Research process started for session: {session.session_id}")
            print(f"Current state: {session.state}")
        
        elif args.command == 'continue-research':
            session = await commands.load_session(args.session_id, args.data_dir)
            coordinator = Coordinator(session)
            await commands.continue_research(coordinator, args.feedback)
            print(f"Research process continued for session: {session.session_id}")
            print(f"Current state: {session.state}")
        
        elif args.command == 'get-meta-review':
            session = await commands.load_session(args.session_id, args.data_dir)
            coordinator = Coordinator(session)
            meta_review = await commands.get_meta_review(coordinator)
            await commands.print_meta_review(meta_review)
        
        elif args.command == 'session-status':
            session = await commands.load_session(args.session_id, args.data_dir)
            await commands.print_session_status(session)
        
        elif args.command == 'print-hypotheses':
            session = await commands.load_session(args.session_id, args.data_dir)
            await commands.print_hypotheses(session, args.iteration)
        
        elif args.command == 'print-rankings':
            session = await commands.load_session(args.session_id, args.data_dir)
            await commands.print_rankings(session, args.iteration)
        
        elif args.command == 'print-reflections':
            session = await commands.load_session(args.session_id, args.data_dir)
            await commands.print_reflections(session, args.iteration)
        
        elif args.command == 'print-evolutions':
            session = await commands.load_session(args.session_id, args.data_dir)
            await commands.print_evolutions(session, args.iteration)
        
        else:
            print(f"Unknown command: {args.command}")
            parser.print_help()
    
    except Exception as e:
        logger.error(f"Error executing command: {e}")
        raise


def main():
    """Main entry point."""
    asyncio.run(main_async())


if __name__ == '__main__':
    main() 