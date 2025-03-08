"""
Cross-domain synthesizer for integrating knowledge across multiple scientific domains.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Set, Tuple

from .knowledge_manager import DomainKnowledgeManager

logger = logging.getLogger("co_scientist")

class CrossDomainSynthesizer:
    """
    Cross-domain synthesizer for integrating knowledge across multiple scientific domains.
    
    This class orchestrates the collection and integration of information from different
    domain-specific databases and knowledge sources, synthesizing a comprehensive
    view that spans multiple scientific fields.
    """
    
    def __init__(self, knowledge_manager: DomainKnowledgeManager):
        """
        Initialize the cross-domain synthesizer.
        
        Args:
            knowledge_manager (DomainKnowledgeManager): The domain knowledge manager
        """
        self.knowledge_manager = knowledge_manager
        
        # Define cross-domain relationships for interdisciplinary connections
        self.domain_relationships = {
            "biomedicine": ["biology", "chemistry"],
            "biology": ["biomedicine", "chemistry"],
            "chemistry": ["biology", "physics"],
            "physics": ["chemistry", "mathematics"],
            "computer_science": ["mathematics", "physics"],
            "mathematics": ["computer_science", "physics"]
        }
        
        # Map of research topics to relevant domains and keywords
        self.topic_domain_map = {
            "drug_discovery": {
                "domains": ["biomedicine", "chemistry", "biology"],
                "keywords": ["drug", "target", "binding", "therapeutic", "compound", "pharmacology"]
            },
            "protein_structure": {
                "domains": ["biology", "chemistry", "physics"],
                "keywords": ["protein", "structure", "folding", "conformation", "domain", "binding site"]
            },
            "gene_regulation": {
                "domains": ["biology", "biomedicine"],
                "keywords": ["gene", "regulation", "expression", "transcription", "promoter", "enhancer"]
            },
            "medical_imaging": {
                "domains": ["biomedicine", "physics", "computer_science"],
                "keywords": ["imaging", "mri", "ct scan", "ultrasound", "image processing", "visualization"]
            },
            "bioinformatics": {
                "domains": ["biology", "computer_science", "mathematics"],
                "keywords": ["algorithm", "sequence", "alignment", "database", "genome", "mutation"]
            },
            "nanomaterials": {
                "domains": ["chemistry", "physics", "biology"],
                "keywords": ["nanoparticle", "nanomaterial", "synthesis", "characterization", "application", "surface"]
            },
            "machine_learning": {
                "domains": ["computer_science", "mathematics", "biomedicine"],
                "keywords": ["algorithm", "neural network", "deep learning", "classification", "regression", "feature"]
            },
            "quantum_computing": {
                "domains": ["physics", "computer_science", "mathematics"],
                "keywords": ["qubit", "quantum", "entanglement", "superposition", "algorithm", "gate"]
            }
        }
    
    def detect_research_domains(self, text: str) -> List[Tuple[str, float]]:
        """
        Detect relevant scientific domains for a research question.
        
        Args:
            text (str): The research question or topic text
            
        Returns:
            List[Tuple[str, float]]: List of (domain, relevance_score) tuples
        """
        text_lower = text.lower()
        domain_scores = {}
        
        # Define domain-specific keywords with weights
        domain_keywords = {
            "biomedicine": {
                "disease": 1.0, "drug": 1.0, "therapy": 1.0, "patient": 1.0, "clinical": 1.0,
                "medicine": 1.0, "biomarker": 0.9, "diagnostic": 0.9, "treatment": 0.9, 
                "pathology": 0.9, "therapeutic": 0.8, "medical": 0.8, "cancer": 0.8,
                "infection": 0.7, "mortality": 0.7
            },
            "biology": {
                "gene": 1.0, "protein": 1.0, "cell": 1.0, "organism": 1.0, "molecular": 1.0,
                "enzyme": 0.9, "receptor": 0.9, "signaling": 0.9, "pathway": 0.9, 
                "expression": 0.8, "mutation": 0.8, "tissue": 0.8, "biological": 0.7,
                "cellular": 0.7, "genome": 0.7, "metabolic": 0.7
            },
            "chemistry": {
                "compound": 1.0, "reaction": 1.0, "synthesis": 1.0, "molecule": 1.0, "chemical": 1.0,
                "polymer": 0.9, "catalyst": 0.9, "ligand": 0.9, "structure": 0.8, 
                "bond": 0.8, "organic": 0.8, "inorganic": 0.8, "spectroscopy": 0.7,
                "crystallography": 0.7, "solvent": 0.7
            },
            "physics": {
                "particle": 1.0, "quantum": 1.0, "energy": 1.0, "force": 1.0, "field": 1.0,
                "mechanics": 0.9, "relativity": 0.9, "gravity": 0.9, "nuclear": 0.9, 
                "electromagnetic": 0.8, "radiation": 0.8, "optics": 0.8, "thermodynamics": 0.8,
                "photon": 0.7, "electron": 0.7, "wave": 0.7
            },
            "computer_science": {
                "algorithm": 1.0, "computation": 1.0, "network": 1.0, "software": 1.0, "system": 1.0,
                "data": 0.9, "learning": 0.9, "artificial intelligence": 0.9, "machine learning": 0.9, 
                "optimization": 0.8, "database": 0.8, "neural network": 0.8, "programming": 0.7,
                "computational": 0.7, "security": 0.7
            },
            "mathematics": {
                "theorem": 1.0, "proof": 1.0, "equation": 1.0, "mathematics": 1.0, "mathematical": 1.0,
                "algebra": 0.9, "geometry": 0.9, "topology": 0.9, "calculus": 0.9, 
                "probability": 0.8, "statistics": 0.8, "numerical": 0.8, "differential": 0.7,
                "function": 0.7, "optimization": 0.7
            }
        }
        
        # Score each domain based on keyword matches
        for domain, keywords in domain_keywords.items():
            score = 0.0
            matches = 0
            
            for keyword, weight in keywords.items():
                if keyword in text_lower:
                    score += weight
                    matches += 1
            
            # Normalize score by number of matches if there are any
            if matches > 0:
                domain_scores[domain] = score / matches
        
        # Check for interdisciplinary topics
        for topic, topic_info in self.topic_domain_map.items():
            topic_matches = sum(1 for keyword in topic_info["keywords"] if keyword in text_lower)
            
            if topic_matches >= 2:  # If at least 2 keywords match, consider this topic relevant
                # Boost scores for domains related to this topic
                topic_score = topic_matches / len(topic_info["keywords"])
                for domain in topic_info["domains"]:
                    if domain in domain_scores:
                        domain_scores[domain] = max(domain_scores[domain], topic_score)
                    else:
                        domain_scores[domain] = topic_score * 0.8  # Slightly lower weight if not already detected
        
        # Sort domains by score and return
        sorted_domains = sorted([(domain, score) for domain, score in domain_scores.items()],
                               key=lambda x: x[1], reverse=True)
        
        return sorted_domains
    
    async def search_across_domains(self, query: str, domains: List[str] = None, limit_per_domain: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search for information across multiple domains.
        
        Args:
            query (str): The search query
            domains (List[str], optional): Specific domains to search. Defaults to None.
            limit_per_domain (int, optional): Maximum results per domain. Defaults to 5.
            
        Returns:
            Dict[str, List[Dict[str, Any]]]: Dict mapping domain names to search results
        """
        # If domains not specified, detect them from the query
        if not domains:
            detected_domains = self.detect_research_domains(query)
            if detected_domains:
                # Use top 3 domains or all domains with score > 0.5
                domains = [domain for domain, score in detected_domains if score > 0.5]
                domains = domains[:3]  # Limit to top 3
        
        # Search across specified domains
        results = await self.knowledge_manager.search(query, domains=domains, limit=limit_per_domain)
        return results
    
    async def get_entity_with_related_cross_domain(self, entity_id: str, provider_id: str, 
                                           related_limit: int = 5, cross_domain_limit: int = 3) -> Dict[str, Any]:
        """
        Get an entity with related entities and cross-domain connections.
        
        Args:
            entity_id (str): ID of the entity to retrieve
            provider_id (str): ID of the provider to use
            related_limit (int, optional): Maximum related entities. Defaults to 5.
            cross_domain_limit (int, optional): Maximum cross-domain connections. Defaults to 3.
            
        Returns:
            Dict[str, Any]: Entity with related and cross-domain information
        """
        # Get the main entity
        entity = await self.knowledge_manager.get_entity(entity_id, provider_id=provider_id)
        if not entity:
            return {}
        
        # Determine the entity's domain
        entity_domain = None
        provider = self.knowledge_manager.get_provider(provider_id)
        if provider:
            entity_domain = provider.domain
        
        # Get related entities in the same domain
        related_entities = await self.knowledge_manager.get_related_entities(
            entity_id, provider_id=provider_id, limit=related_limit
        )
        
        # Get cross-domain connections
        cross_domain_info = {}
        if entity_domain:
            # Define which domains to search for cross-domain connections
            cross_domains = self.domain_relationships.get(entity_domain, [])
            
            # Extract keywords from the entity for cross-domain search
            keywords = self._extract_entity_keywords(entity)
            
            for domain in cross_domains:
                if keywords:
                    # Create a search query using entity keywords
                    cross_domain_query = " ".join(keywords[:3])  # Use top 3 keywords
                    
                    # Search this domain for related concepts
                    domain_results = await self.knowledge_manager._search_domain(
                        domain, cross_domain_query, limit=cross_domain_limit
                    )
                    
                    if domain_results:
                        cross_domain_info[domain] = domain_results
        
        # Combine all information
        result = {
            "entity": entity,
            "related_entities": related_entities,
            "cross_domain_connections": cross_domain_info
        }
        
        return result
    
    def _extract_entity_keywords(self, entity: Dict[str, Any]) -> List[str]:
        """
        Extract keywords from an entity for cross-domain searching.
        
        Args:
            entity (Dict[str, Any]): The entity to extract keywords from
            
        Returns:
            List[str]: List of keywords
        """
        keywords = set()
        
        # Add title terms
        if "title" in entity:
            title_terms = [t for t in entity["title"].split() if len(t) > 3]
            keywords.update(title_terms)
        
        # Add specific fields based on entity type
        if entity.get("provider") == "pubmed" or entity.get("provider") == "arxiv":
            # For publications, use title, abstract, and keywords
            if "abstract" in entity:
                # Extract noun phrases and technical terms from abstract
                abstract_words = entity["abstract"].split()
                keywords.update([w for w in abstract_words if len(w) > 5])  # Longer words tend to be technical terms
        
        elif entity.get("provider") == "uniprot":
            # For proteins, use protein name, gene names, and function
            if "protein_name" in entity:
                keywords.add(entity["protein_name"])
            
            if "gene_names" in entity:
                keywords.update(entity["gene_names"])
            
            if "function" in entity:
                function_words = entity["function"].split()
                keywords.update([w for w in function_words if len(w) > 5])
        
        elif entity.get("provider") == "pubchem":
            # For chemicals, use name, formula, and synonyms
            if "name" in entity:
                keywords.add(entity["name"])
            
            if "formula" in entity:
                keywords.add(entity["formula"])
            
            if "synonyms" in entity:
                for synonym in entity["synonyms"][:5]:  # Use top 5 synonyms
                    keywords.add(synonym)
        
        # Remove punctuation and convert to lowercase
        cleaned_keywords = []
        for keyword in keywords:
            # Remove common punctuation
            cleaned = keyword.strip(".,;:()[]{}\"'").lower()
            if cleaned and len(cleaned) > 3:
                cleaned_keywords.append(cleaned)
        
        # Sort by length (longer terms likely more specific)
        cleaned_keywords.sort(key=len, reverse=True)
        
        return cleaned_keywords[:10]  # Return top 10 keywords
    
    async def synthesize_multi_domain_knowledge(self, query: str, 
                                         max_entities_per_domain: int = 3, 
                                         max_related_per_entity: int = 3) -> Dict[str, Any]:
        """
        Synthesize knowledge across multiple domains for a research query.
        
        Args:
            query (str): The research query
            max_entities_per_domain (int, optional): Max entities per domain. Defaults to 3.
            max_related_per_entity (int, optional): Max related entities per entity. Defaults to 3.
            
        Returns:
            Dict[str, Any]: Synthesized knowledge across domains
        """
        # Detect relevant domains for this query
        detected_domains = self.detect_research_domains(query)
        primary_domains = [domain for domain, score in detected_domains if score > 0.5][:3]
        
        if not primary_domains:
            # If no specific domains detected, use general purpose domains
            primary_domains = ["biomedicine", "biology", "chemistry"]
        
        # Search across all primary domains
        domain_results = await self.knowledge_manager.search(
            query, domains=primary_domains, limit=max_entities_per_domain
        )
        
        # For each top entity in each domain, get related entities and cross-domain connections
        enriched_results = {}
        
        for domain, results in domain_results.items():
            domain_enriched = []
            
            for result in results:
                provider_id = result.get("provider")
                entity_id = result.get("id")
                
                if provider_id and entity_id:
                    # Get entity with related and cross-domain information
                    enriched_entity = await self.get_entity_with_related_cross_domain(
                        entity_id, 
                        provider_id, 
                        related_limit=max_related_per_entity,
                        cross_domain_limit=max_related_per_entity
                    )
                    
                    if enriched_entity and "entity" in enriched_entity:
                        domain_enriched.append(enriched_entity)
            
            if domain_enriched:
                enriched_results[domain] = domain_enriched
        
        # Add metadata about the synthesis
        synthesis = {
            "query": query,
            "primary_domains": primary_domains,
            "domain_relevance": {domain: score for domain, score in detected_domains},
            "results": enriched_results
        }
        
        return synthesis
    
    def format_synthesis_highlights(self, synthesis: Dict[str, Any], max_highlights_per_domain: int = 3) -> str:
        """
        Format the key highlights from a knowledge synthesis.
        
        Args:
            synthesis (Dict[str, Any]): The knowledge synthesis result
            max_highlights_per_domain (int, optional): Maximum highlights per domain. Defaults to 3.
            
        Returns:
            str: Formatted highlights
        """
        if not synthesis or "results" not in synthesis:
            return "No synthesis results available."
        
        query = synthesis.get("query", "")
        primary_domains = synthesis.get("primary_domains", [])
        
        highlights = [f"# Knowledge Synthesis for: {query}", ""]
        
        # Domain overview
        highlights.append("## Domains Overview")
        for domain, score in synthesis.get("domain_relevance", {}).items():
            if score > 0.3:  # Only show reasonably relevant domains
                highlights.append(f"- {domain.capitalize()}: Relevance score {score:.2f}")
        highlights.append("")
        
        # Key findings by domain
        for domain in primary_domains:
            domain_results = synthesis.get("results", {}).get(domain, [])
            if domain_results:
                highlights.append(f"## {domain.capitalize()} Domain Findings")
                
                # Add top entities from this domain
                for i, result in enumerate(domain_results[:max_highlights_per_domain]):
                    entity = result.get("entity", {})
                    title = entity.get("title", "Unknown")
                    source = entity.get("source", entity.get("provider", "unknown"))
                    
                    highlights.append(f"### {i+1}. {title}")
                    
                    # Add entity description or summary
                    if entity.get("abstract"):
                        # For publications, show a condensed abstract
                        abstract = entity["abstract"]
                        if len(abstract) > 250:
                            abstract = abstract[:250] + "..."
                        highlights.append(f"{abstract}")
                    elif entity.get("function"):
                        # For proteins, show function
                        highlights.append(f"{entity['function']}")
                    elif entity.get("description"):
                        # For other entities with descriptions
                        highlights.append(f"{entity['description']}")
                    
                    # Add source
                    highlights.append(f"Source: {source}")
                    
                    # Add cross-domain connections if available
                    cross_domain = result.get("cross_domain_connections", {})
                    if cross_domain:
                        highlights.append("#### Cross-domain connections:")
                        for cd_domain, cd_entities in cross_domain.items():
                            cd_titles = [e.get("title", "Unknown") for e in cd_entities[:2]]
                            highlights.append(f"- {cd_domain.capitalize()}: {', '.join(cd_titles)}")
                    
                    highlights.append("")
        
        # Add a synthesis section
        highlights.append("## Synthesis Across Domains")
        highlights.append("The findings suggest connections between:")
        
        # Look for connection patterns across domains
        domain_keywords = {}
        for domain, results in synthesis.get("results", {}).items():
            domain_terms = set()
            for result in results:
                entity = result.get("entity", {})
                keywords = self._extract_entity_keywords(entity)
                domain_terms.update(keywords)
            domain_keywords[domain] = domain_terms
        
        # Find overlapping terms between domains
        if len(domain_keywords) >= 2:
            domains = list(domain_keywords.keys())
            for i in range(len(domains)):
                for j in range(i+1, len(domains)):
                    domain1 = domains[i]
                    domain2 = domains[j]
                    
                    common_terms = domain_keywords[domain1].intersection(domain_keywords[domain2])
                    if common_terms:
                        top_common = list(common_terms)[:3]
                        highlights.append(f"- {domain1.capitalize()} and {domain2.capitalize()}: {', '.join(top_common)}")
        
        return "\n".join(highlights)
    
    async def get_cross_domain_document_context(self, query: str, max_documents_per_domain: int = 2) -> Dict[str, str]:
        """
        Get document content from multiple domains to serve as context for LLM reasoning.
        
        Args:
            query (str): The research query
            max_documents_per_domain (int, optional): Max documents per domain. Defaults to 2.
            
        Returns:
            Dict[str, str]: Domain to document content mapping
        """
        # Detect domains from query
        detected_domains = self.detect_research_domains(query)
        domains = [domain for domain, score in detected_domains if score > 0.4][:3]
        
        if not domains:
            domains = ["biomedicine", "biology", "chemistry"]  # Default domains
        
        # Search across domains
        domain_results = await self.knowledge_manager.search(query, domains=domains, limit=max_documents_per_domain)
        
        # Extract full documents/entities for context
        domain_documents = {}
        
        for domain, results in domain_results.items():
            domain_content = []
            
            for result in results:
                provider_id = result.get("provider")
                entity_id = result.get("id")
                
                if provider_id and entity_id:
                    # Get full entity content
                    entity = await self.knowledge_manager.get_entity(entity_id, provider_id=provider_id)
                    
                    if entity:
                        # Format entity as readable text
                        content = f"--- {entity.get('title', 'Unknown Title')} ---\n"
                        
                        if entity.get("abstract"):
                            content += f"Abstract: {entity['abstract']}\n"
                        
                        if entity.get("authors"):
                            content += f"Authors: {', '.join(entity['authors'])}\n"
                        
                        if entity.get("function"):
                            content += f"Function: {entity['function']}\n"
                        
                        if entity.get("description"):
                            content += f"Description: {entity['description']}\n"
                        
                        # Add specific entity type content
                        if provider_id == "uniprot":
                            if entity.get("gene_names"):
                                content += f"Genes: {', '.join(entity['gene_names'])}\n"
                            
                            if entity.get("organism"):
                                content += f"Organism: {entity['organism']}\n"
                        
                        elif provider_id == "pubchem":
                            if entity.get("formula"):
                                content += f"Formula: {entity['formula']}\n"
                            
                            if entity.get("synonyms"):
                                content += f"Synonyms: {', '.join(entity['synonyms'][:5])}\n"
                        
                        # Add source reference
                        content += f"Source: {provider_id}, ID: {entity_id}\n\n"
                        
                        domain_content.append(content)
            
            if domain_content:
                domain_documents[domain] = "\n".join(domain_content)
        
        return domain_documents