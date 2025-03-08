"""
Domain knowledge manager for accessing and coordinating domain-specific knowledge providers.
"""

import os
import logging
import importlib
import asyncio
from typing import Dict, List, Any, Optional, Union, Type, Tuple

from .base_provider import DomainKnowledgeProvider
from .pubmed_provider import PubMedProvider

# Import new providers
from .providers.arxiv_provider import ArxivProvider
from .providers.pubchem_provider import PubChemProvider
from .providers.uniprot_provider import UniProtProvider

logger = logging.getLogger("co_scientist")

class DomainKnowledgeManager:
    """
    Manager for domain-specific knowledge providers.
    
    This class manages multiple domain knowledge providers, making it easier to access
    domain-specific information across different scientific domains. It handles provider
    initialization, selection, and coordination for comprehensive scientific database integration.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the domain knowledge manager.
        
        Args:
            config (Dict[str, Any], optional): Configuration for domain providers. Defaults to None.
        """
        self.config = config or {}
        self.providers: Dict[str, DomainKnowledgeProvider] = {}
        
        # Provider specializations (what each provider is good at)
        self.provider_specializations: Dict[str, Dict[str, float]] = {}
        
        # Domain to provider mappings (primary and secondary providers for each domain)
        self.domain_provider_map: Dict[str, List[str]] = {
            "biomedicine": ["pubmed", "uniprot"],
            "biology": ["uniprot", "pubmed"],
            "chemistry": ["pubchem", "pubmed"],
            "physics": ["arxiv", "pubmed"],
            "computer_science": ["arxiv", "pubmed"],
            "mathematics": ["arxiv", "pubmed"],
            "multi_domain": ["arxiv", "pubmed"],
        }
        
        # Initialize domain map with all provider domains
        self.initialized = False
        
        # Register built-in providers
        self.register_builtin_providers()
    
    def register_builtin_providers(self):
        """Register the built-in domain knowledge providers."""
        # PubMed for biomedicine
        if os.environ.get("NCBI_API_KEY") or self.config.get("pubmed", {}).get("api_key"):
            self.register_provider("pubmed", PubMedProvider, self.config.get("pubmed", {}))
            self.provider_specializations["pubmed"] = {
                "literature": 1.0,
                "citations": 0.9,
                "biomedicine": 0.9,
                "biology": 0.8,
                "chemistry": 0.6,
                "multi_domain": 0.7
            }
            logger.info("Registered PubMed provider for biomedicine domain")
        
        # ArXiv for physics, computer science, mathematics
        self.register_provider("arxiv", ArxivProvider, self.config.get("arxiv", {}))
        self.provider_specializations["arxiv"] = {
            "literature": 1.0,
            "citations": 0.8,
            "physics": 0.9,
            "computer_science": 0.9,
            "mathematics": 0.9,
            "biology": 0.3,
            "chemistry": 0.3,
            "multi_domain": 0.8
        }
        logger.info("Registered ArXiv provider for physics, computer science, and mathematics domains")
        
        # PubChem for chemistry
        self.register_provider("pubchem", PubChemProvider, self.config.get("pubchem", {}))
        self.provider_specializations["pubchem"] = {
            "chemistry": 1.0,
            "compounds": 1.0,
            "structures": 0.9,
            "assays": 0.8,
            "biology": 0.5,
            "multi_domain": 0.6
        }
        logger.info("Registered PubChem provider for chemistry domain")
        
        # UniProt for biology
        self.register_provider("uniprot", UniProtProvider, self.config.get("uniprot", {}))
        self.provider_specializations["uniprot"] = {
            "proteins": 1.0,
            "genes": 0.9,
            "biology": 1.0,
            "biomedicine": 0.8,
            "multi_domain": 0.5
        }
        logger.info("Registered UniProt provider for biology domain")
    
    def register_provider(self, provider_id: str, provider_class: Type[DomainKnowledgeProvider], config: Dict[str, Any] = None):
        """
        Register a domain knowledge provider.
        
        Args:
            provider_id (str): The provider identifier (e.g., 'pubmed', 'arxiv')
            provider_class (Type[DomainKnowledgeProvider]): The provider class
            config (Dict[str, Any], optional): Configuration for the provider. Defaults to None.
        """
        try:
            provider = provider_class(config)
            self.providers[provider_id.lower()] = provider
            logger.info(f"Registered {provider_class.__name__} provider with ID {provider_id}")
        except Exception as e:
            logger.error(f"Error registering provider {provider_id}: {str(e)}")
    
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
            
        # Determine which providers to initialize based on requested domains
        providers_to_init = set()
        if domains:
            for domain in domains:
                domain = domain.lower()
                if domain in self.domain_provider_map:
                    providers_to_init.update(self.domain_provider_map[domain])
                else:
                    # If domain not explicitly mapped, try to initialize all providers
                    providers_to_init = set(self.providers.keys())
                    break
        else:
            # Initialize all providers if no domains specified
            providers_to_init = set(self.providers.keys())
        
        # Initialize providers concurrently
        init_tasks = []
        for provider_id in providers_to_init:
            if provider_id in self.providers:
                provider = self.providers[provider_id]
                init_tasks.append(self._init_provider(provider_id, provider))
        
        # Wait for all initialization tasks to complete
        results = await asyncio.gather(*init_tasks, return_exceptions=True)
        
        # Check if at least one provider was successfully initialized
        successful = any(isinstance(result, bool) and result for result in results)
        self.initialized = successful
        
        return successful
    
    async def _init_provider(self, provider_id: str, provider: DomainKnowledgeProvider) -> bool:
        """Helper method to initialize a provider and handle exceptions."""
        try:
            result = await provider.initialize()
            if result:
                logger.info(f"Successfully initialized provider {provider_id}")
                return True
            else:
                logger.warning(f"Failed to initialize provider {provider_id}")
                return False
        except Exception as e:
            logger.error(f"Error initializing provider {provider_id}: {str(e)}")
            return False
    
    def get_provider(self, provider_id: str) -> Optional[DomainKnowledgeProvider]:
        """
        Get a specific provider by ID.
        
        Args:
            provider_id (str): The provider ID
            
        Returns:
            Optional[DomainKnowledgeProvider]: The provider or None if not found
        """
        return self.providers.get(provider_id.lower())
    
    def get_providers_for_domain(self, domain: str) -> List[Tuple[str, DomainKnowledgeProvider]]:
        """
        Get all providers that can handle a specific domain, sorted by relevance.
        
        Args:
            domain (str): The domain name
            
        Returns:
            List[Tuple[str, DomainKnowledgeProvider]]: List of (provider_id, provider) tuples
        """
        domain = domain.lower()
        providers = []
        
        # First check the domain map
        if domain in self.domain_provider_map:
            for provider_id in self.domain_provider_map[domain]:
                if provider_id in self.providers:
                    providers.append((provider_id, self.providers[provider_id]))
        
        # If no providers found in the map, check specializations
        if not providers:
            provider_scores = []
            for provider_id, provider in self.providers.items():
                specializations = self.provider_specializations.get(provider_id, {})
                score = specializations.get(domain, 0.0)
                if score > 0:
                    provider_scores.append((provider_id, provider, score))
            
            # Sort by score and extract providers
            provider_scores.sort(key=lambda x: x[2], reverse=True)
            providers = [(p_id, p) for p_id, p, _ in provider_scores]
        
        # If still no providers, return all providers
        if not providers:
            providers = list(self.providers.items())
        
        return providers
    
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
            await self.initialize(domains)
            
        results = {}
        
        # Determine which domains to search
        domains_to_search = [d.lower() for d in domains] if domains else list(self.domain_provider_map.keys())
        
        # Create search tasks for each domain
        search_tasks = []
        for domain in domains_to_search:
            task = self._search_domain(domain, query, limit)
            search_tasks.append(task)
        
        # Wait for all search tasks to complete
        domain_results = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        # Process results
        for domain, domain_result in zip(domains_to_search, domain_results):
            if isinstance(domain_result, Exception):
                logger.error(f"Error searching {domain} domain: {str(domain_result)}")
                results[domain] = []
            else:
                results[domain] = domain_result
        
        return results
    
    async def _search_domain(self, domain: str, query: str, limit: int) -> List[Dict[str, Any]]:
        """Helper method to search a domain using appropriate providers."""
        providers = self.get_providers_for_domain(domain)
        if not providers:
            return []
        
        combined_results = []
        provider_limit = limit  # Initially request full limit from each provider
        
        for provider_id, provider in providers:
            try:
                # Search using this provider
                provider_results = await provider.search(query, limit=provider_limit)
                
                # Add provider metadata to results
                for result in provider_results:
                    if not result.get("provider"):
                        result["provider"] = provider_id
                
                # Add results to combined list
                combined_results.extend(provider_results)
                
                # If we got enough results, reduce limit for next provider
                if len(combined_results) >= limit:
                    break
                
                # Adjust limit for next provider
                provider_limit = limit - len(combined_results)
                
            except Exception as e:
                logger.error(f"Error searching with provider {provider_id}: {str(e)}")
        
        # Sort results by relevance (if provider added a score) and limit to requested number
        combined_results = combined_results[:limit]
        
        return combined_results
    
    async def multi_domain_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Perform a comprehensive search across all domains and combine results.
        
        Args:
            query (str): The search query
            limit (int, optional): Maximum total results. Defaults to 10.
            
        Returns:
            List[Dict[str, Any]]: Combined search results from all domains
        """
        # Search all domains concurrently
        results = await self.search(query, domains=None, limit=limit)
        
        # Combine and deduplicate results
        combined_results = []
        seen_ids = set()
        
        # Prioritize results from each domain
        for domain, domain_results in results.items():
            for result in domain_results:
                result_id = f"{result.get('provider', 'unknown')}:{result.get('id', '')}"
                if result_id not in seen_ids:
                    seen_ids.add(result_id)
                    combined_results.append(result)
                    
                    if len(combined_results) >= limit:
                        return combined_results
        
        return combined_results
    
    async def get_entity(self, entity_id: str, provider_id: str = None, domain: str = None) -> Optional[Dict[str, Any]]:
        """
        Retrieve detailed information about a specific entity.
        
        Args:
            entity_id (str): ID of the entity
            provider_id (str, optional): Specific provider to use. Defaults to None.
            domain (str, optional): Domain to use if provider not specified. Defaults to None.
            
        Returns:
            Optional[Dict[str, Any]]: Entity information or None if not found
        """
        if not self.initialized:
            await self.initialize()
        
        # Case 1: Specific provider requested
        if provider_id:
            provider = self.get_provider(provider_id)
            if provider:
                try:
                    return await provider.get_entity(entity_id)
                except Exception as e:
                    logger.error(f"Error getting entity {entity_id} from provider {provider_id}: {str(e)}")
            return None
        
        # Case 2: Domain specified, try providers for that domain
        if domain:
            providers = self.get_providers_for_domain(domain)
            for provider_id, provider in providers:
                try:
                    entity = await provider.get_entity(entity_id)
                    if entity:
                        return entity
                except Exception as e:
                    logger.error(f"Error getting entity {entity_id} from provider {provider_id}: {str(e)}")
            return None
        
        # Case 3: No provider or domain specified, try all providers
        for provider_id, provider in self.providers.items():
            try:
                entity = await provider.get_entity(entity_id)
                if entity:
                    return entity
            except Exception as e:
                logger.debug(f"Provider {provider_id} could not get entity {entity_id}: {str(e)}")
        
        return None
    
    async def get_related_entities(self, entity_id: str, provider_id: str = None, domain: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get entities related to the specified entity.
        
        Args:
            entity_id (str): ID of the entity
            provider_id (str, optional): Specific provider to use. Defaults to None.
            domain (str, optional): Domain to use if provider not specified. Defaults to None.
            limit (int, optional): Maximum number of results. Defaults to 10.
            
        Returns:
            List[Dict[str, Any]]: List of related entities
        """
        if not self.initialized:
            await self.initialize()
        
        # Case 1: Specific provider requested
        if provider_id:
            provider = self.get_provider(provider_id)
            if provider:
                try:
                    return await provider.get_related_entities(entity_id, limit=limit)
                except Exception as e:
                    logger.error(f"Error getting related entities for {entity_id} from provider {provider_id}: {str(e)}")
            return []
        
        # Case 2: Domain specified, try providers for that domain
        if domain:
            providers = self.get_providers_for_domain(domain)
            for provider_id, provider in providers:
                try:
                    related = await provider.get_related_entities(entity_id, limit=limit)
                    if related:
                        return related
                except Exception as e:
                    logger.error(f"Error getting related entities for {entity_id} from provider {provider_id}: {str(e)}")
            return []
        
        # Case 3: No provider or domain specified
        # First determine which provider knows about this entity
        entity_provider = None
        entity_domain = None
        
        for provider_id, provider in self.providers.items():
            try:
                entity = await provider.get_entity(entity_id)
                if entity:
                    entity_provider = provider_id
                    # Determine domain from provider specializations
                    specializations = self.provider_specializations.get(provider_id, {})
                    # Find domain with highest specialization score
                    if specializations:
                        entity_domain = max(
                            (d for d in specializations.keys() if d in self.domain_provider_map),
                            key=lambda d: specializations.get(d, 0),
                            default=None
                        )
                    break
            except Exception:
                pass
        
        # If we found a provider that knows this entity, use it
        if entity_provider:
            provider = self.get_provider(entity_provider)
            try:
                return await provider.get_related_entities(entity_id, limit=limit)
            except Exception as e:
                logger.error(f"Error getting related entities for {entity_id} from provider {entity_provider}: {str(e)}")
        
        # If we found a domain but no results from the primary provider, try other providers for that domain
        if entity_domain:
            providers = self.get_providers_for_domain(entity_domain)
            for p_id, provider in providers:
                if p_id != entity_provider:  # Skip the provider we already tried
                    try:
                        related = await provider.get_related_entities(entity_id, limit=limit)
                        if related:
                            return related
                    except Exception:
                        pass
        
        return []
    
    def format_citation(self, entity: Dict[str, Any], style: str = "apa") -> str:
        """
        Format citation for an entity according to the specified citation style.
        
        Args:
            entity (Dict[str, Any]): The entity to cite
            style (str, optional): Citation style. Defaults to "apa".
            
        Returns:
            str: Formatted citation string
        """
        # Determine which provider to use for citation
        provider_id = entity.get("provider")
        if provider_id and provider_id in self.providers:
            provider = self.providers[provider_id]
            try:
                return provider.format_citation(entity, style=style)
            except Exception as e:
                logger.error(f"Error formatting citation with provider {provider_id}: {str(e)}")
        
        # If provider not found, use domain
        domain = entity.get("domain")
        if domain:
            providers = self.get_providers_for_domain(domain)
            for p_id, provider in providers:
                try:
                    return provider.format_citation(entity, style=style)
                except Exception:
                    pass
        
        # Fallback citation format
        authors = ", ".join(entity.get("authors", []))
        title = entity.get("title", "")
        year = entity.get("year", "")
        source = entity.get("journal", entity.get("source", ""))
        
        if not authors and not source:
            # Minimal citation with just title
            return f"{title} ({year})."
        
        return f"{authors or 'Unknown'} ({year or 'n.d.'}). {title}. {source or 'Unknown source'}."