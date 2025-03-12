"""
Base Agent interface for mini-RAUL.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


class Agent(ABC):
    """Base agent class that all agent types must implement."""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the agent.
        
        Args:
            name: The name of the agent.
            config: Optional configuration for the agent.
        """
        self.name = name
        self.config = config or {}
    
    @abstractmethod
    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent's task.
        
        Args:
            context: The context information for the agent to operate with.
            
        Returns:
            Dict containing the results of the agent's execution.
        """
        pass
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})" 