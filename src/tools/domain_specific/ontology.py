"""
Domain ontology management for the Co-Scientist system.
"""

import os
import json
import logging
from typing import Dict, List, Set, Any, Optional, Tuple

logger = logging.getLogger("co_scientist")

class DomainOntology:
    """
    Manages domain-specific ontologies for scientific fields.
    
    This class handles loading, navigating, and utilizing domain-specific scientific
    ontologies, which provide structured representations of domain concepts and their
    relationships. These ontologies help ground hypotheses in established domain
    knowledge and terminology.
    """
    
    def __init__(self, domain: str, ontology_file: Optional[str] = None):
        """
        Initialize a domain ontology.
        
        Args:
            domain (str): The domain name (e.g., 'biology', 'chemistry')
            ontology_file (Optional[str], optional): Path to ontology file. If None, looks for
                domain-specific file in the default ontology directory. Defaults to None.
        """
        self.domain = domain.lower()
        self.ontology_file = ontology_file
        self.concepts: Dict[str, Dict[str, Any]] = {}
        self.relationships: Dict[str, List[Tuple[str, str, Dict[str, Any]]]] = {}
        self.initialized = False
        
        # Default ontology directory (within the package)
        self.ontology_dir = os.path.join(os.path.dirname(__file__), "ontologies")
        
        # Create the ontology directory if it doesn't exist
        os.makedirs(self.ontology_dir, exist_ok=True)
    
    def initialize(self) -> bool:
        """
        Load the ontology from file.
        
        Returns:
            bool: True if ontology loaded successfully, False otherwise
        """
        try:
            # Determine ontology file path
            file_path = self.ontology_file
            if not file_path:
                file_path = os.path.join(self.ontology_dir, f"{self.domain}_ontology.json")
            
            # Check if file exists
            if not os.path.exists(file_path):
                logger.warning(f"Ontology file not found: {file_path}")
                return False
            
            # Load ontology from file
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Parse concepts
            self.concepts = data.get("concepts", {})
            
            # Parse relationships
            self._build_relationships()
            
            self.initialized = True
            logger.info(f"Loaded {self.domain} ontology with {len(self.concepts)} concepts")
            return True
            
        except Exception as e:
            logger.error(f"Error loading {self.domain} ontology: {str(e)}")
            return False
    
    def _build_relationships(self):
        """Build the relationships index from concepts."""
        self.relationships = {}
        
        # Iterate through all concepts
        for concept_id, concept_data in self.concepts.items():
            # Process each relationship in this concept
            for rel_type, related_concepts in concept_data.get("relationships", {}).items():
                if not isinstance(related_concepts, list):
                    related_concepts = [related_concepts]
                
                # Add this relationship to the index
                if concept_id not in self.relationships:
                    self.relationships[concept_id] = []
                
                for related in related_concepts:
                    if isinstance(related, dict):
                        related_id = related.get("id")
                        metadata = {k: v for k, v in related.items() if k != "id"}
                    else:
                        related_id = related
                        metadata = {}
                    
                    if related_id:
                        self.relationships[concept_id].append((rel_type, related_id, metadata))
    
    def get_concept(self, concept_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a concept by ID.
        
        Args:
            concept_id (str): The concept ID
            
        Returns:
            Optional[Dict[str, Any]]: The concept data or None if not found
        """
        if not self.initialized:
            if not self.initialize():
                return None
        
        return self.concepts.get(concept_id)
    
    def search_concepts(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for concepts by name or description.
        
        Args:
            query (str): The search query
            limit (int, optional): Maximum number of results. Defaults to 10.
            
        Returns:
            List[Dict[str, Any]]: Matching concepts
        """
        if not self.initialized:
            if not self.initialize():
                return []
        
        query = query.lower()
        results = []
        
        for concept_id, concept_data in self.concepts.items():
            name = concept_data.get("name", "").lower()
            description = concept_data.get("description", "").lower()
            synonyms = [s.lower() for s in concept_data.get("synonyms", [])]
            
            # Check for matches
            if (query in name or 
                query in description or 
                any(query in synonym for synonym in synonyms)):
                
                results.append({
                    "id": concept_id,
                    **concept_data
                })
                
                if len(results) >= limit:
                    break
        
        return results
    
    def get_related_concepts(self, concept_id: str, relation_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get concepts related to the specified concept.
        
        Args:
            concept_id (str): The concept ID
            relation_type (Optional[str], optional): Filter by relationship type. Defaults to None.
            
        Returns:
            List[Dict[str, Any]]: Related concepts
        """
        if not self.initialized:
            if not self.initialize():
                return []
        
        if concept_id not in self.relationships:
            return []
        
        results = []
        
        for rel_type, related_id, metadata in self.relationships[concept_id]:
            if relation_type is None or rel_type == relation_type:
                related_concept = self.get_concept(related_id)
                if related_concept:
                    results.append({
                        "id": related_id,
                        "relation_type": rel_type,
                        "metadata": metadata,
                        **related_concept
                    })
        
        return results
    
    def validate_term(self, term: str) -> Tuple[bool, Optional[str], List[Dict[str, Any]]]:
        """
        Validate if a term matches a known concept in the ontology.
        
        Args:
            term (str): The term to validate
            
        Returns:
            Tuple[bool, Optional[str], List[Dict[str, Any]]]: 
                (is_valid, normalized_term, suggested_alternatives)
        """
        if not self.initialized:
            if not self.initialize():
                return False, None, []
        
        term_lower = term.lower()
        
        # Check direct matches
        for concept_id, concept_data in self.concepts.items():
            name = concept_data.get("name", "").lower()
            synonyms = [s.lower() for s in concept_data.get("synonyms", [])]
            
            if term_lower == name or term_lower in synonyms:
                return True, concept_data.get("name"), []
        
        # No direct match, find similar concepts as suggestions
        suggestions = self.search_concepts(term, limit=5)
        return False, None, suggestions
    
    def get_domain_hierarchy(self) -> Dict[str, Any]:
        """
        Get the domain concept hierarchy.
        
        Returns:
            Dict[str, Any]: Hierarchical representation of the domain concepts
        """
        if not self.initialized:
            if not self.initialize():
                return {}
        
        # Find root concepts (those without 'is_a' relationships to other concepts)
        root_concepts = set(self.concepts.keys())
        
        for concept_id, relationships in self.relationships.items():
            for rel_type, related_id, _ in relationships:
                if rel_type == "is_a" and related_id in root_concepts:
                    root_concepts.discard(concept_id)
        
        # Build hierarchy
        hierarchy = {}
        
        for root in root_concepts:
            hierarchy[root] = self._build_concept_tree(root)
            
        return hierarchy
    
    def _build_concept_tree(self, concept_id: str) -> Dict[str, Any]:
        """Recursively build a concept subtree."""
        concept = self.get_concept(concept_id)
        if not concept:
            return {}
            
        # Create node
        node = {
            "id": concept_id,
            "name": concept.get("name", ""),
            "children": {}
        }
        
        # Find children (concepts that have 'is_a' relationship to this one)
        for child_id, relationships in self.relationships.items():
            for rel_type, related_id, _ in relationships:
                if rel_type == "is_a" and related_id == concept_id:
                    # Recursively build child subtree
                    node["children"][child_id] = self._build_concept_tree(child_id)
        
        return node