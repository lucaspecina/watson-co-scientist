"""
Base class for domain-specific knowledge providers.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union

logger = logging.getLogger("co_scientist")

class DomainKnowledgeProvider(ABC):
    """
    Abstract base class for domain-specific knowledge providers.
    
    This class defines the interface that all domain knowledge providers must implement.
    Domain knowledge providers connect to external databases, ontologies, or APIs
    to retrieve specialized knowledge for specific scientific domains.
    """
    
    def __init__(self, domain: str, config: Dict[str, Any] = None):
        """
        Initialize the domain knowledge provider.
        
        Args:
            domain (str): The scientific domain this provider specializes in (e.g., 'biology', 'chemistry')
            config (Dict[str, Any], optional): Configuration for the provider. Defaults to None.
        """
        self.domain = domain
        self.config = config or {}
        self._is_initialized = False
        logger.info(f"Initializing {domain} domain knowledge provider")
    
    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the provider, connecting to any external resources.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search the domain knowledge base.
        
        Args:
            query (str): The search query
            limit (int, optional): Maximum number of results. Defaults to 10.
            
        Returns:
            List[Dict[str, Any]]: List of search results
        """
        pass
    
    @abstractmethod
    async def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve detailed information about a specific entity.
        
        Args:
            entity_id (str): ID of the entity
            
        Returns:
            Optional[Dict[str, Any]]: Entity information or None if not found
        """
        pass
    
    @abstractmethod
    async def get_related_entities(self, entity_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get entities related to the specified entity.
        
        Args:
            entity_id (str): ID of the entity
            limit (int, optional): Maximum number of results. Defaults to 10.
            
        Returns:
            List[Dict[str, Any]]: List of related entities
        """
        pass
    
    @abstractmethod
    def format_citation(self, entity: Dict[str, Any], style: str = "apa") -> str:
        """
        Format citation for an entity according to the specified citation style.
        
        Args:
            entity (Dict[str, Any]): The entity to cite
            style (str, optional): Citation style (e.g., 'apa', 'mla', 'chicago'). Defaults to "apa".
            
        Returns:
            str: Formatted citation string
        """
        pass
    
    @property
    def is_initialized(self) -> bool:
        """Check if the provider is initialized."""
        return self._is_initialized