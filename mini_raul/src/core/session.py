"""
Session management for mini-RAUL.
"""
import json
import os
import time
import logging
from enum import Enum
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import uuid

logger = logging.getLogger(__name__)


class SessionState(Enum):
    """Possible states for a research session."""
    
    CREATED = "created"                  # Session created but not started
    PLANNING = "planning"                # Planning the research
    GENERATING = "generating"            # Generating hypotheses
    EVALUATING = "evaluating"            # Evaluating hypotheses
    RANKING = "ranking"                  # Ranking hypotheses
    REFLECTING = "reflecting"            # Reflecting on hypotheses
    EVOLVING = "evolving"                # Evolving hypotheses
    WAITING_FEEDBACK = "waiting_feedback"  # Waiting for user feedback
    COMPLETED = "completed"              # Session completed
    FAILED = "failed"                    # Session failed


class Session:
    """
    Research session management.
    Handles state, storage and context for a research session.
    """
    
    def __init__(self, 
                session_id: Optional[str] = None, 
                data_dir: Optional[str] = None):
        """
        Initialize a research session.
        
        Args:
            session_id: Unique identifier for the session.
            data_dir: Directory to store session data.
        """
        self.session_id = session_id or str(uuid.uuid4())
        self.data_dir = data_dir or os.path.join("data", "sessions")
        self.session_dir = os.path.join(self.data_dir, self.session_id)
        
        # Initialize session data structure
        self.data = {
            "session_id": self.session_id,
            "created_at": time.time(),
            "updated_at": time.time(),
            "state": SessionState.CREATED.value,
            "research_goal": "",
            "iterations": [],
            "current_iteration": None,
            "metadata": {}
        }
        
        # Create session directory if it doesn't exist
        if not os.path.exists(self.session_dir):
            os.makedirs(self.session_dir, exist_ok=True)
    
    def save(self) -> None:
        """Save the session to disk."""
        self.data["updated_at"] = time.time()
        
        with open(os.path.join(self.session_dir, "session.json"), "w") as f:
            json.dump(self.data, f, indent=2)
    
    @classmethod
    def load(cls, session_id: str, data_dir: Optional[str] = None) -> "Session":
        """
        Load an existing session.
        
        Args:
            session_id: ID of the session to load.
            data_dir: Directory containing session data.
            
        Returns:
            The loaded session.
            
        Raises:
            FileNotFoundError: If the session doesn't exist.
        """
        session = cls(session_id, data_dir)
        
        if not os.path.exists(os.path.join(session.session_dir, "session.json")):
            raise FileNotFoundError(f"Session {session_id} not found")
        
        with open(os.path.join(session.session_dir, "session.json"), "r") as f:
            session.data = json.load(f)
        
        return session
    
    @property
    def state(self) -> str:
        """Get the current state of the session."""
        return self.data["state"]
    
    @state.setter
    def state(self, state: Union[str, SessionState]) -> None:
        """Set the state of the session."""
        if isinstance(state, SessionState):
            self.data["state"] = state.value
        else:
            self.data["state"] = state
        
        self.save()
    
    @property
    def research_goal(self) -> str:
        """Get the research goal."""
        return self.data["research_goal"]
    
    @research_goal.setter
    def research_goal(self, goal: str) -> None:
        """Set the research goal."""
        self.data["research_goal"] = goal
        self.save()
    
    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the session."""
        self.data["metadata"][key] = value
        self.save()
    
    def start_iteration(self) -> int:
        """
        Start a new iteration.
        
        Returns:
            The iteration number.
        """
        iteration = {
            "iteration_number": len(self.data["iterations"]) + 1,
            "started_at": time.time(),
            "completed_at": None,
            "hypotheses": [],
            "rankings": [],
            "reflections": [],
            "evolution": [],
            "user_feedback": "",
            "state": SessionState.PLANNING.value
        }
        
        self.data["iterations"].append(iteration)
        self.data["current_iteration"] = len(self.data["iterations"]) - 1
        self.save()
        
        return iteration["iteration_number"]
    
    @property
    def current_iteration(self) -> Optional[Dict[str, Any]]:
        """Get the current iteration if one exists."""
        if self.data["current_iteration"] is not None:
            return self.data["iterations"][self.data["current_iteration"]]
        return None
    
    def add_hypothesis(self, 
                      hypothesis: Dict[str, Any], 
                      agent_name: str) -> None:
        """
        Add a hypothesis to the current iteration.
        
        Args:
            hypothesis: The hypothesis data.
            agent_name: Name of the agent that generated the hypothesis.
        """
        if not self.current_iteration:
            raise ValueError("No active iteration")
        
        hypothesis_data = {
            "id": str(uuid.uuid4()),
            "content": hypothesis,
            "agent": agent_name,
            "created_at": time.time(),
            "metadata": {}
        }
        
        self.current_iteration["hypotheses"].append(hypothesis_data)
        self.save()
    
    def add_ranking(self, 
                   rankings: List[Dict[str, Any]], 
                   agent_name: str) -> None:
        """
        Add ranking results to the current iteration.
        
        Args:
            rankings: The ranking data.
            agent_name: Name of the agent that generated the ranking.
        """
        if not self.current_iteration:
            raise ValueError("No active iteration")
        
        ranking_data = {
            "id": str(uuid.uuid4()),
            "rankings": rankings,
            "agent": agent_name,
            "created_at": time.time(),
            "metadata": {}
        }
        
        self.current_iteration["rankings"].append(ranking_data)
        self.save()
    
    def add_reflection(self, 
                      reflection: Dict[str, Any], 
                      agent_name: str) -> None:
        """
        Add a reflection to the current iteration.
        
        Args:
            reflection: The reflection data.
            agent_name: Name of the agent that generated the reflection.
        """
        if not self.current_iteration:
            raise ValueError("No active iteration")
        
        reflection_data = {
            "id": str(uuid.uuid4()),
            "content": reflection,
            "agent": agent_name,
            "created_at": time.time(),
            "metadata": {}
        }
        
        self.current_iteration["reflections"].append(reflection_data)
        self.save()
    
    def add_evolution(self, 
                     evolution: Dict[str, Any], 
                     agent_name: str) -> None:
        """
        Add an evolution to the current iteration.
        
        Args:
            evolution: The evolution data.
            agent_name: Name of the agent that generated the evolution.
        """
        if not self.current_iteration:
            raise ValueError("No active iteration")
        
        evolution_data = {
            "id": str(uuid.uuid4()),
            "content": evolution,
            "agent": agent_name,
            "created_at": time.time(),
            "metadata": {}
        }
        
        self.current_iteration["evolution"].append(evolution_data)
        self.save()
    
    def add_user_feedback(self, feedback: str) -> None:
        """
        Add user feedback to the current iteration.
        
        Args:
            feedback: The user feedback.
        """
        if not self.current_iteration:
            raise ValueError("No active iteration")
        
        self.current_iteration["user_feedback"] = feedback
        self.save()
    
    def complete_iteration(self) -> None:
        """Mark the current iteration as completed."""
        if not self.current_iteration:
            raise ValueError("No active iteration")
        
        self.current_iteration["completed_at"] = time.time()
        self.current_iteration["state"] = SessionState.COMPLETED.value
        self.save()
    
    @classmethod
    def list_sessions(cls, data_dir: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all available sessions.
        
        Args:
            data_dir: Directory containing session data.
            
        Returns:
            List of session metadata.
        """
        data_dir = data_dir or os.path.join("data", "sessions")
        
        if not os.path.exists(data_dir):
            return []
        
        sessions = []
        
        for session_id in os.listdir(data_dir):
            session_file = os.path.join(data_dir, session_id, "session.json")
            
            if os.path.exists(session_file):
                try:
                    with open(session_file, "r") as f:
                        data = json.load(f)
                    
                    sessions.append({
                        "session_id": data["session_id"],
                        "created_at": data["created_at"],
                        "updated_at": data["updated_at"],
                        "state": data["state"],
                        "research_goal": data["research_goal"],
                        "iteration_count": len(data["iterations"])
                    })
                except Exception as e:
                    logger.error(f"Error loading session {session_id}: {e}")
        
        return sorted(sessions, key=lambda s: s["updated_at"], reverse=True) 