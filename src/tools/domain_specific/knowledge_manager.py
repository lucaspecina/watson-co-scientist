"""
Domain knowledge manager for accessing and coordinating domain-specific knowledge providers.
"""

import os
import logging
import importlib
from typing import Dict, List, Any, Optional, Union, Type

from .base_provider import DomainKnowledgeProvider
from .pubmed_provider import PubMedProvider

logger = logging.getLogger("co_scientist")

class DomainKnowledgeManager:
    """
    Manager for domain-specific knowledge providers.
    
    This class manages multiple domain knowledge providers, making it easier to access
    domain-specific information across different scientific domains. It handles provider
    initialization, selection, and coordination.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the domain knowledge manager.
        
        Args:
            config (Dict[str, Any], optional): Configuration for domain providers. Defaults to None.
        """
        self.config = config or {}
        self.providers: Dict[str, DomainKnowledgeProvider] = {}
        self.initialized = False
        
        # Register built-in providers
        self.register_builtin_providers()
    
    def register_builtin_providers(self):
        """Register the built-in domain knowledge providers."""
        # PubMed for biomedicine
        if os.environ.get("NCBI_API_KEY") or self.config.get("pubmed", {}).get("api_key"):
            self.register_provider("biomedicine", PubMedProvider, self.config.get("pubmed", {}))
            logger.info("Registered PubMed provider for biomedicine domain")
    
    def register_provider(self, domain: str, provider_class: Type[DomainKnowledgeProvider], config: Dict[str, Any] = None):
        """
        Register a domain knowledge provider.
        
        Args:
            domain (str): The domain name (e.g., 'biomedicine', 'chemistry')
            provider_class (Type[DomainKnowledgeProvider]): The provider class
            config (Dict[str, Any], optional): Configuration for the provider. Defaults to None.
        """
        try:
            provider = provider_class(config)
            self.providers[domain.lower()] = provider
            logger.info(f"Registered {provider_class.__name__} for {domain} domain")
        except Exception as e:
            logger.error(f"Error registering provider for {domain} domain: {str(e)}")
    
    async def initialize(self, domains: List[str] = None) -> bool:
        """
        Initialize the registered providers.
        
        Args:
            domains (List[str], optional): List of domains to initialize. If None, initializes all. Defaults to None.
            
        Returns:
            bool: True if at least one provider was successfully initialized, False otherwise
        """
        if not self.providers:
            logger.warning("No domain knowledge providers registered")
            return False
            
        # Initialize requested domains or all domains
        domains_to_init = [d.lower() for d in domains] if domains else list(self.providers.keys())
        
        successful = False
        for domain, provider in self.providers.items():
            if domain in domains_to_init:
                try:
                    result = await provider.initialize()
                    if result:
                        logger.info(f"Successfully initialized {domain} provider")
                        successful = True
                    else:
                        logger.warning(f"Failed to initialize {domain} provider")
                except Exception as e:
                    logger.error(f"Error initializing {domain} provider: {str(e)}")
        
        self.initialized = successful
        return successful
    
    def get_provider(self, domain: str) -> Optional[DomainKnowledgeProvider]:
        """
        Get a specific domain knowledge provider.
        
        Args:
            domain (str): The domain name
            
        Returns:
            Optional[DomainKnowledgeProvider]: The provider or None if not found
        """
        return self.providers.get(domain.lower())
    
    async def search(self, query: str, domains: List[str] = None, limit: int = 10) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search across multiple domain knowledge bases.
        
        Args:
            query (str): The search query
            domains (List[str], optional): Domains to search. If None, searches all. Defaults to None.
            limit (int, optional): Maximum results per domain. Defaults to 10.
            
        Returns:
            Dict[str, List[Dict[str, Any]]]: Dict mapping domain names to search results
        """
        if not self.initialized:
            await self.initialize()
            
        results = {}
        
        # Determine which domains to search
        domains_to_search = [d.lower() for d in domains] if domains else list(self.providers.keys())
        
        # Perform search across all specified domains
        for domain, provider in self.providers.items():
            if domain in domains_to_search:
                try:
                    domain_results = await provider.search(query, limit=limit)
                    results[domain] = domain_results
                except Exception as e:
                    logger.error(f"Error searching {domain} domain: {str(e)}")
                    results[domain] = []
        
        return results
    
    async def get_entity(self, domain: str, entity_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve detailed information about a specific entity.
        
        Args:
            domain (str): The domain name
            entity_id (str): ID of the entity
            
        Returns:
            Optional[Dict[str, Any]]: Entity information or None if not found
        """
        if not self.initialized:
            await self.initialize()
            
        provider = self.get_provider(domain)
        if provider:
            try:
                return await provider.get_entity(entity_id)
            except Exception as e:
                logger.error(f"Error getting entity {entity_id} from {domain} domain: {str(e)}")
        
        return None
    
    async def get_related_entities(self, domain: str, entity_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get entities related to the specified entity.
        
        Args:
            domain (str): The domain name
            entity_id (str): ID of the entity
            limit (int, optional): Maximum number of results. Defaults to 10.
            
        Returns:
            List[Dict[str, Any]]: List of related entities
        """
        if not self.initialized:
            await self.initialize()
            
        provider = self.get_provider(domain)
        if provider:
            try:
                return await provider.get_related_entities(entity_id, limit=limit)
            except Exception as e:
                logger.error(f"Error getting related entities for {entity_id} from {domain} domain: {str(e)}")
        
        return []
    
    def format_citation(self, domain: str, entity: Dict[str, Any], style: str = "apa") -> str:
        """
        Format citation for an entity according to the specified citation style.
        
        Args:
            domain (str): The domain name
            entity (Dict[str, Any]): The entity to cite
            style (str, optional): Citation style. Defaults to "apa".
            
        Returns:
            str: Formatted citation string
        """
        provider = self.get_provider(domain)
        if provider:
            try:
                return provider.format_citation(entity, style=style)
            except Exception as e:
                logger.error(f"Error formatting citation in {domain} domain: {str(e)}")
        
        # Fallback citation format
        authors = ", ".join(entity.get("authors", []))
        title = entity.get("title", "")
        year = entity.get("year", "")
        source = entity.get("journal", entity.get("source", ""))
        
        return f"{authors} ({year}). {title}. {source}."