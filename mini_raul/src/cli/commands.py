"""
Command-line interface commands for mini-RAUL.
"""
import os
import sys
import json
import logging
import asyncio
from typing import Dict, Any, List, Optional, Union

from ..core.session import Session, SessionState
from ..core.coordinator import Coordinator

logger = logging.getLogger(__name__)


async def create_session(session_id: str, data_dir: Optional[str] = None) -> Session:
    """
    Create a new research session.
    
    Args:
        session_id: ID for the new session.
        data_dir: Directory to store session data.
        
    Returns:
        The created session.
    """
    session = Session(session_id, data_dir)
    session.save()
    
    logger.info(f"Created new session: {session_id}")
    return session


async def list_sessions(data_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    List all available sessions.
    
    Args:
        data_dir: Directory containing session data.
        
    Returns:
        List of session metadata.
    """
    sessions = Session.list_sessions(data_dir)
    
    if not sessions:
        logger.info("No sessions found")
    else:
        logger.info(f"Found {len(sessions)} sessions")
    
    return sessions


async def load_session(session_id: str, data_dir: Optional[str] = None) -> Session:
    """
    Load an existing session.
    
    Args:
        session_id: ID of the session to load.
        data_dir: Directory containing session data.
        
    Returns:
        The loaded session.
    """
    try:
        session = Session.load(session_id, data_dir)
        logger.info(f"Loaded session: {session_id}")
        return session
    except FileNotFoundError:
        logger.error(f"Session not found: {session_id}")
        raise


async def start_research(coordinator: Coordinator, research_goal: str, preferences: str = "", constraints: str = "") -> None:
    """
    Start a new research process.
    
    Args:
        coordinator: The coordinator to use.
        research_goal: The research goal.
        preferences: User preferences.
        constraints: User constraints.
    """
    await coordinator.start_session(research_goal, preferences, constraints)
    logger.info("Research process started")


async def continue_research(coordinator: Coordinator, feedback: str = "") -> None:
    """
    Continue an existing research process with a new iteration.
    
    Args:
        coordinator: The coordinator to use.
        feedback: User feedback for the previous iteration.
    """
    if coordinator.session.state != SessionState.WAITING_FEEDBACK.value:
        logger.error(f"Session is not waiting for feedback. Current state: {coordinator.session.state}")
        return
    
    # Add user feedback if provided
    if feedback:
        await coordinator.add_user_feedback(feedback)
    
    # Complete the current iteration
    coordinator.session.complete_iteration()
    
    # Start a new iteration
    coordinator.session.start_iteration()
    
    # Run the new iteration
    await coordinator.run_iteration()
    
    logger.info("Research process continued with a new iteration")


async def get_meta_review(coordinator: Coordinator) -> Dict[str, Any]:
    """
    Get a meta-review of the research progress.
    
    Args:
        coordinator: The coordinator to use.
        
    Returns:
        The meta-review.
    """
    meta_review = await coordinator.create_meta_review()
    logger.info("Meta-review created")
    return meta_review


async def print_session_status(session: Session) -> None:
    """
    Print the status of a session.
    
    Args:
        session: The session to print status for.
    """
    print(f"Session ID: {session.session_id}")
    print(f"State: {session.state}")
    print(f"Research Goal: {session.research_goal}")
    print(f"Iterations: {len(session.data['iterations'])}")
    
    if session.current_iteration:
        print("\nCurrent Iteration:")
        print(f"  Number: {session.current_iteration['iteration_number']}")
        print(f"  State: {session.current_iteration['state']}")
        print(f"  Hypotheses: {len(session.current_iteration['hypotheses'])}")
        print(f"  Rankings: {len(session.current_iteration['rankings'])}")
        print(f"  Reflections: {len(session.current_iteration['reflections'])}")
        print(f"  Evolutions: {len(session.current_iteration['evolution'])}")
        
        if session.current_iteration['user_feedback']:
            print(f"  User Feedback: {session.current_iteration['user_feedback'][:100]}...")


async def print_hypotheses(session: Session, iteration: Optional[int] = None) -> None:
    """
    Print the hypotheses for a session.
    
    Args:
        session: The session to print hypotheses for.
        iteration: Optional iteration number to print hypotheses for.
    """
    if iteration is not None:
        if iteration < 1 or iteration > len(session.data['iterations']):
            logger.error(f"Invalid iteration number: {iteration}")
            return
        
        iteration_data = session.data['iterations'][iteration - 1]
    else:
        iteration_data = session.current_iteration
    
    if not iteration_data:
        logger.error("No active iteration")
        return
    
    print(f"Hypotheses for Iteration {iteration_data['iteration_number']}:")
    
    for i, hypothesis in enumerate(iteration_data['hypotheses']):
        print(f"\nHypothesis {i+1} (ID: {hypothesis['id']}):")
        print(f"  Agent: {hypothesis['agent']}")
        
        content = hypothesis.get('content', {})
        hypothesis_data = content.get('hypothesis', {})
        
        if isinstance(hypothesis_data, dict):
            for key, value in hypothesis_data.items():
                if isinstance(value, str):
                    # Truncate long strings
                    if len(value) > 100:
                        value = value[:100] + "..."
                print(f"  {key}: {value}")
        else:
            print(f"  Content: {str(hypothesis_data)[:100]}...")


async def print_rankings(session: Session, iteration: Optional[int] = None) -> None:
    """
    Print the rankings for a session.
    
    Args:
        session: The session to print rankings for.
        iteration: Optional iteration number to print rankings for.
    """
    if iteration is not None:
        if iteration < 1 or iteration > len(session.data['iterations']):
            logger.error(f"Invalid iteration number: {iteration}")
            return
        
        iteration_data = session.data['iterations'][iteration - 1]
    else:
        iteration_data = session.current_iteration
    
    if not iteration_data:
        logger.error("No active iteration")
        return
    
    if not iteration_data['rankings']:
        print(f"No rankings for Iteration {iteration_data['iteration_number']}")
        return
    
    print(f"Rankings for Iteration {iteration_data['iteration_number']}:")
    
    for i, ranking_data in enumerate(iteration_data['rankings']):
        print(f"\nRanking {i+1} (Agent: {ranking_data['agent']}):")
        
        rankings = ranking_data.get('rankings', [])
        
        # Sort by rank
        sorted_rankings = sorted(rankings, key=lambda x: x.get('rank', 999))
        
        for rank in sorted_rankings:
            print(f"  Rank {rank.get('rank', 'N/A')}: Hypothesis {rank.get('id', 'N/A')} (Wins: {rank.get('wins', 0)}, Losses: {rank.get('losses', 0)})")


async def print_reflections(session: Session, iteration: Optional[int] = None) -> None:
    """
    Print the reflections for a session.
    
    Args:
        session: The session to print reflections for.
        iteration: Optional iteration number to print reflections for.
    """
    if iteration is not None:
        if iteration < 1 or iteration > len(session.data['iterations']):
            logger.error(f"Invalid iteration number: {iteration}")
            return
        
        iteration_data = session.data['iterations'][iteration - 1]
    else:
        iteration_data = session.current_iteration
    
    if not iteration_data:
        logger.error("No active iteration")
        return
    
    if not iteration_data['reflections']:
        print(f"No reflections for Iteration {iteration_data['iteration_number']}")
        return
    
    print(f"Reflections for Iteration {iteration_data['iteration_number']}:")
    
    for i, reflection in enumerate(iteration_data['reflections']):
        print(f"\nReflection {i+1} (Agent: {reflection['agent']}):")
        
        content = reflection.get('content', {})
        reflection_data = content.get('reflection', {})
        reflection_type = content.get('reflection_type', 'unknown')
        
        print(f"  Type: {reflection_type}")
        
        if isinstance(reflection_data, dict):
            for key, value in reflection_data.items():
                if isinstance(value, str):
                    # Truncate long strings
                    if len(value) > 100:
                        value = value[:100] + "..."
                print(f"  {key}: {value}")
        else:
            print(f"  Content: {str(reflection_data)[:100]}...")


async def print_evolutions(session: Session, iteration: Optional[int] = None) -> None:
    """
    Print the evolutions for a session.
    
    Args:
        session: The session to print evolutions for.
        iteration: Optional iteration number to print evolutions for.
    """
    if iteration is not None:
        if iteration < 1 or iteration > len(session.data['iterations']):
            logger.error(f"Invalid iteration number: {iteration}")
            return
        
        iteration_data = session.data['iterations'][iteration - 1]
    else:
        iteration_data = session.current_iteration
    
    if not iteration_data:
        logger.error("No active iteration")
        return
    
    if not iteration_data['evolution']:
        print(f"No evolutions for Iteration {iteration_data['iteration_number']}")
        return
    
    print(f"Evolutions for Iteration {iteration_data['iteration_number']}:")
    
    for i, evolution in enumerate(iteration_data['evolution']):
        print(f"\nEvolution {i+1} (Agent: {evolution['agent']}):")
        
        content = evolution.get('content', {})
        evolution_data = content.get('evolved_hypothesis', {})
        evolution_type = content.get('evolution_type', 'unknown')
        
        print(f"  Type: {evolution_type}")
        
        if isinstance(evolution_data, dict):
            for key, value in evolution_data.items():
                if isinstance(value, str):
                    # Truncate long strings
                    if len(value) > 100:
                        value = value[:100] + "..."
                print(f"  {key}: {value}")
        else:
            print(f"  Content: {str(evolution_data)[:100]}...")


async def print_meta_review(meta_review: Dict[str, Any]) -> None:
    """
    Print a meta-review.
    
    Args:
        meta_review: The meta-review to print.
    """
    print("Meta-Review:")
    
    content = meta_review.get('meta_review', {})
    
    if isinstance(content, dict):
        for key, value in content.items():
            if isinstance(value, str):
                # Truncate long strings
                if len(value) > 100:
                    value = value[:100] + "..."
            print(f"  {key}: {value}")
    else:
        print(f"  Content: {str(content)[:100]}...") 