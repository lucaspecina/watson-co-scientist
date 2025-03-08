"""
Knowledge graph for scientific papers.

This module provides a graph-based representation of scientific knowledge
extracted from papers, allowing for relationship traversal, querying, and synthesis.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field, asdict
from collections import defaultdict

logger = logging.getLogger("co_scientist")

@dataclass
class Entity:
    """Represents an entity node in the knowledge graph."""
    id: str  # Unique identifier
    name: str  # Entity name/term
    type: str  # Entity type (e.g., "scientific_concept", "methodology", etc.)
    definition: str  # Entity definition
    importance: float = 0.0  # Importance score (0-10)
    papers: List[str] = field(default_factory=list)  # Papers mentioning this entity
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

@dataclass
class Relation:
    """Represents a relationship edge in the knowledge graph."""
    id: str  # Unique identifier
    source_id: str  # Source entity ID
    target_id: str  # Target entity ID
    type: str  # Relation type (e.g., "causes", "correlates_with", etc.)
    evidence: str  # Evidence for this relationship
    confidence: float = 0.0  # Confidence score (0-10)
    papers: List[str] = field(default_factory=list)  # Papers mentioning this relation
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

@dataclass
class Paper:
    """Represents a scientific paper in the knowledge graph."""
    id: str  # Unique identifier
    title: str  # Paper title
    authors: List[str]  # Paper authors
    abstract: str  # Paper abstract
    year: str = ""  # Publication year
    url: str = ""  # URL to the paper
    pdf_path: str = ""  # Path to local PDF
    extraction_path: str = ""  # Path to extraction JSON
    knowledge_path: str = ""  # Path to knowledge JSON
    entities: List[str] = field(default_factory=list)  # Entity IDs in this paper
    relations: List[str] = field(default_factory=list)  # Relation IDs in this paper
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

class KnowledgeGraph:
    """
    Graph-based representation of scientific knowledge.
    
    This class provides a unified graph of entities and relationships extracted
    from scientific papers, with methods for traversal, querying, and synthesis.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the knowledge graph.
        
        Args:
            config (Dict[str, Any], optional): Configuration dictionary. Defaults to None.
        """
        self.config = config or {}
        
        # Configure storage directory
        self.storage_dir = self.config.get('storage_dir', 'data/knowledge_graph')
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Initialize graph data structures
        self.entities: Dict[str, Entity] = {}  # Map entity ID to Entity
        self.relations: Dict[str, Relation] = {}  # Map relation ID to Relation
        self.papers: Dict[str, Paper] = {}  # Map paper ID to Paper
        
        # Create index data structures
        self.entity_name_to_id: Dict[str, str] = {}  # Map entity name to ID
        self.entity_type_index: Dict[str, Set[str]] = defaultdict(set)  # Map entity type to set of entity IDs
        self.relation_type_index: Dict[str, Set[str]] = defaultdict(set)  # Map relation type to set of relation IDs
        self.paper_entity_index: Dict[str, Set[str]] = defaultdict(set)  # Map paper ID to set of entity IDs
        self.paper_relation_index: Dict[str, Set[str]] = defaultdict(set)  # Map paper ID to set of relation IDs
        self.entity_paper_index: Dict[str, Set[str]] = defaultdict(set)  # Map entity ID to set of paper IDs
        self.entity_relation_index: Dict[str, Set[str]] = defaultdict(set)  # Map entity ID to set of relation IDs
    
    def add_paper_knowledge(self, paper_id: str, knowledge: Dict[str, Any]) -> None:
        """
        Add knowledge extracted from a paper to the graph.
        
        Args:
            paper_id (str): Unique paper ID
            knowledge (Dict[str, Any]): Knowledge extracted from the paper
        """
        try:
            # Extract basic paper information
            title = knowledge.get("title", "Unknown Title")
            authors = knowledge.get("authors", [])
            abstract = knowledge.get("source_paper", {}).get("abstract", "")
            year = knowledge.get("year", "")
            
            # Create or update paper node
            paper = Paper(
                id=paper_id,
                title=title,
                authors=authors,
                abstract=abstract,
                year=year
            )
            
            # Add paths if available
            source_paper = knowledge.get("source_paper", {})
            paper.pdf_path = source_paper.get("pdf_path", "")
            paper.extraction_path = knowledge.get("extraction_path", "")
            paper.knowledge_path = knowledge.get("knowledge_path", "")
            
            # Process entities
            entities = knowledge.get("entities", [])
            for entity_data in entities:
                # Extract entity data
                entity_name = entity_data.get("name", "Unknown Entity")
                entity_type = entity_data.get("type", "unknown")
                entity_def = entity_data.get("definition", "")
                importance = float(entity_data.get("importance_score", 0))
                
                # Generate entity ID (either use existing or create new)
                entity_id = self._find_or_create_entity_id(entity_name, entity_type, entity_def)
                
                # Add entity ID to paper
                paper.entities.append(entity_id)
                
                # Create or update entity node
                if entity_id not in self.entities:
                    entity = Entity(
                        id=entity_id,
                        name=entity_name,
                        type=entity_type,
                        definition=entity_def,
                        importance=importance,
                        papers=[paper_id]
                    )
                    self.entities[entity_id] = entity
                    
                    # Update indexes
                    self.entity_name_to_id[entity_name.lower()] = entity_id
                    self.entity_type_index[entity_type].add(entity_id)
                else:
                    # Update existing entity
                    entity = self.entities[entity_id]
                    if paper_id not in entity.papers:
                        entity.papers.append(paper_id)
                    
                    # Update importance if higher
                    if importance > entity.importance:
                        entity.importance = importance
                
                # Update paper-entity index
                self.paper_entity_index[paper_id].add(entity_id)
                self.entity_paper_index[entity_id].add(paper_id)
            
            # Process relations
            relations = knowledge.get("relations", [])
            for relation_data in relations:
                # Extract relation data
                source_name = relation_data.get("source", "")
                target_name = relation_data.get("target", "")
                relation_type = relation_data.get("relation", "unknown")
                evidence = relation_data.get("evidence", "")
                confidence = float(relation_data.get("confidence", 0))
                
                # Skip if source or target missing
                if not source_name or not target_name:
                    continue
                
                # Find entity IDs
                source_id = self.entity_name_to_id.get(source_name.lower())
                target_id = self.entity_name_to_id.get(target_name.lower())
                
                # Skip if entities not found
                if not source_id or not target_id:
                    continue
                
                # Generate relation ID
                relation_id = f"{source_id}|{relation_type}|{target_id}"
                
                # Add relation ID to paper
                paper.relations.append(relation_id)
                
                # Create or update relation
                if relation_id not in self.relations:
                    relation = Relation(
                        id=relation_id,
                        source_id=source_id,
                        target_id=target_id,
                        type=relation_type,
                        evidence=evidence,
                        confidence=confidence,
                        papers=[paper_id]
                    )
                    self.relations[relation_id] = relation
                    
                    # Update indexes
                    self.relation_type_index[relation_type].add(relation_id)
                    self.entity_relation_index[source_id].add(relation_id)
                    self.entity_relation_index[target_id].add(relation_id)
                else:
                    # Update existing relation
                    relation = self.relations[relation_id]
                    if paper_id not in relation.papers:
                        relation.papers.append(paper_id)
                    
                    # Update confidence if higher
                    if confidence > relation.confidence:
                        relation.confidence = confidence
                
                # Update paper-relation index
                self.paper_relation_index[paper_id].add(relation_id)
            
            # Save the paper
            self.papers[paper_id] = paper
            
            logger.info(f"Added paper {paper_id} to knowledge graph with {len(paper.entities)} entities and {len(paper.relations)} relations")
            
        except Exception as e:
            logger.error(f"Error adding paper knowledge to graph: {str(e)}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
    
    def _find_or_create_entity_id(self, name: str, entity_type: str, definition: str) -> str:
        """
        Find an existing entity ID or create a new one.
        
        Args:
            name (str): Entity name
            entity_type (str): Entity type
            definition (str): Entity definition
            
        Returns:
            str: Entity ID
        """
        # Check if entity already exists by name
        entity_id = self.entity_name_to_id.get(name.lower())
        if entity_id:
            return entity_id
            
        # Create new entity ID
        entity_id = f"{entity_type}_{len(self.entities)}_{name.lower().replace(' ', '_')}"
        return entity_id
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """
        Get an entity by ID.
        
        Args:
            entity_id (str): Entity ID
            
        Returns:
            Optional[Entity]: The entity or None if not found
        """
        return self.entities.get(entity_id)
    
    def get_relation(self, relation_id: str) -> Optional[Relation]:
        """
        Get a relation by ID.
        
        Args:
            relation_id (str): Relation ID
            
        Returns:
            Optional[Relation]: The relation or None if not found
        """
        return self.relations.get(relation_id)
    
    def get_paper(self, paper_id: str) -> Optional[Paper]:
        """
        Get a paper by ID.
        
        Args:
            paper_id (str): Paper ID
            
        Returns:
            Optional[Paper]: The paper or None if not found
        """
        return self.papers.get(paper_id)
    
    def find_entities_by_name(self, name: str, partial_match: bool = False) -> List[Entity]:
        """
        Find entities by name.
        
        Args:
            name (str): Entity name to search for
            partial_match (bool, optional): Whether to allow partial matches. Defaults to False.
            
        Returns:
            List[Entity]: List of matching entities
        """
        results = []
        
        # Exact match
        entity_id = self.entity_name_to_id.get(name.lower())
        if entity_id and entity_id in self.entities:
            results.append(self.entities[entity_id])
            
        # Partial match if requested
        if partial_match:
            name_lower = name.lower()
            for entity_id, entity in self.entities.items():
                if name_lower in entity.name.lower() and entity not in results:
                    results.append(entity)
        
        return results
    
    def find_entities_by_type(self, entity_type: str) -> List[Entity]:
        """
        Find entities by type.
        
        Args:
            entity_type (str): Entity type to search for
            
        Returns:
            List[Entity]: List of matching entities
        """
        entity_ids = self.entity_type_index.get(entity_type, set())
        return [self.entities[entity_id] for entity_id in entity_ids if entity_id in self.entities]
    
    def find_relations_by_type(self, relation_type: str) -> List[Relation]:
        """
        Find relations by type.
        
        Args:
            relation_type (str): Relation type to search for
            
        Returns:
            List[Relation]: List of matching relations
        """
        relation_ids = self.relation_type_index.get(relation_type, set())
        return [self.relations[relation_id] for relation_id in relation_ids if relation_id in self.relations]
    
    def get_entity_relations(self, entity_id: str) -> List[Relation]:
        """
        Get all relations involving an entity.
        
        Args:
            entity_id (str): Entity ID
            
        Returns:
            List[Relation]: List of relations
        """
        relation_ids = self.entity_relation_index.get(entity_id, set())
        return [self.relations[relation_id] for relation_id in relation_ids if relation_id in self.relations]
    
    def get_entity_papers(self, entity_id: str) -> List[Paper]:
        """
        Get all papers mentioning an entity.
        
        Args:
            entity_id (str): Entity ID
            
        Returns:
            List[Paper]: List of papers
        """
        paper_ids = self.entity_paper_index.get(entity_id, set())
        return [self.papers[paper_id] for paper_id in paper_ids if paper_id in self.papers]
    
    def get_paper_entities(self, paper_id: str) -> List[Entity]:
        """
        Get all entities mentioned in a paper.
        
        Args:
            paper_id (str): Paper ID
            
        Returns:
            List[Entity]: List of entities
        """
        entity_ids = self.paper_entity_index.get(paper_id, set())
        return [self.entities[entity_id] for entity_id in entity_ids if entity_id in self.entities]
    
    def get_paper_relations(self, paper_id: str) -> List[Relation]:
        """
        Get all relations mentioned in a paper.
        
        Args:
            paper_id (str): Paper ID
            
        Returns:
            List[Relation]: List of relations
        """
        relation_ids = self.paper_relation_index.get(paper_id, set())
        return [self.relations[relation_id] for relation_id in relation_ids if relation_id in self.relations]
    
    def find_path(self, source_id: str, target_id: str, max_depth: int = 3) -> List[List[Tuple[str, str, str]]]:
        """
        Find paths between two entities in the graph.
        
        Args:
            source_id (str): Source entity ID
            target_id (str): Target entity ID
            max_depth (int, optional): Maximum path depth. Defaults to 3.
            
        Returns:
            List[List[Tuple[str, str, str]]]: List of paths, each path is a list of (entity_id, relation_type, entity_id) tuples
        """
        # Implementation of breadth-first search to find paths
        visited = set()
        queue = [[(source_id, None, None)]]  # Start with just the source entity
        paths = []
        
        while queue and len(paths) < 10:  # Limit to 10 paths
            path = queue.pop(0)
            node = path[-1][0]  # Last entity in the path
            
            # If we reached the target, add this path to results
            if node == target_id:
                # Filter out the first tuple since it has no relation
                filtered_path = [(s, r, t) for s, r, t in path[1:]]
                paths.append(filtered_path)
                continue
                
            # If we reached max depth, skip
            if len(path) > max_depth:
                continue
                
            # Mark as visited
            visited.add(node)
            
            # Get all relations involving this entity
            relations = self.get_entity_relations(node)
            
            for relation in relations:
                # Determine the other entity in the relation
                other_id = relation.target_id if relation.source_id == node else relation.source_id
                
                # Skip if we've already visited
                if other_id in visited:
                    continue
                    
                # Determine the direction
                if relation.source_id == node:
                    # Outgoing relation
                    next_tuple = (node, relation.type, other_id)
                else:
                    # Incoming relation (reverse the relation)
                    next_tuple = (other_id, f"inverse_{relation.type}", node)
                    
                # Add to queue
                queue.append(path + [next_tuple])
        
        return paths
    
    def find_common_entities(self, paper_ids: List[str]) -> List[Entity]:
        """
        Find entities mentioned in all of the specified papers.
        
        Args:
            paper_ids (List[str]): List of paper IDs
            
        Returns:
            List[Entity]: List of common entities
        """
        if not paper_ids:
            return []
            
        # Get sets of entity IDs for each paper
        paper_entity_sets = [self.paper_entity_index.get(paper_id, set()) for paper_id in paper_ids]
        
        # Find intersection of all sets
        common_entity_ids = set.intersection(*paper_entity_sets)
        
        # Convert to entities
        return [self.entities[entity_id] for entity_id in common_entity_ids if entity_id in self.entities]
    
    def get_entity_neighborhood(self, entity_id: str, max_depth: int = 1) -> Dict[str, Any]:
        """
        Get the neighborhood of an entity up to max_depth.
        
        Args:
            entity_id (str): Entity ID
            max_depth (int, optional): Maximum neighborhood depth. Defaults to 1.
            
        Returns:
            Dict[str, Any]: Neighborhood structure with entities and relations
        """
        neighborhood = {
            "focus_entity": entity_id,
            "entities": {},
            "relations": {}
        }
        
        # Add the focus entity
        entity = self.get_entity(entity_id)
        if not entity:
            return neighborhood
            
        neighborhood["entities"][entity_id] = entity.to_dict()
        
        # BFS to explore neighborhood
        visited = {entity_id}
        current_entities = {entity_id}
        
        for depth in range(max_depth):
            next_entities = set()
            
            # For each entity in the current frontier
            for current_id in current_entities:
                # Get all relations involving this entity
                relations = self.get_entity_relations(current_id)
                
                for relation in relations:
                    # Add relation to neighborhood
                    neighborhood["relations"][relation.id] = relation.to_dict()
                    
                    # Determine the other entity
                    other_id = relation.target_id if relation.source_id == current_id else relation.source_id
                    
                    # Add to next frontier if not visited
                    if other_id not in visited:
                        visited.add(other_id)
                        next_entities.add(other_id)
                        
                        # Add entity to neighborhood
                        other_entity = self.get_entity(other_id)
                        if other_entity:
                            neighborhood["entities"][other_id] = other_entity.to_dict()
            
            # Update current frontier for next iteration
            current_entities = next_entities
            
        return neighborhood
    
    def save(self, path: Optional[str] = None) -> None:
        """
        Save the knowledge graph to a file.
        
        Args:
            path (Optional[str], optional): Path to save to. Defaults to None.
        """
        if not path:
            path = os.path.join(self.storage_dir, "knowledge_graph.json")
            
        data = {
            "entities": {k: v.to_dict() for k, v in self.entities.items()},
            "relations": {k: v.to_dict() for k, v in self.relations.items()},
            "papers": {k: v.to_dict() for k, v in self.papers.items()}
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Saved knowledge graph to {path}")
    
    @classmethod
    def load(cls, path: str, config: Dict[str, Any] = None) -> 'KnowledgeGraph':
        """
        Load a knowledge graph from a file.
        
        Args:
            path (str): Path to load from
            config (Dict[str, Any], optional): Configuration. Defaults to None.
            
        Returns:
            KnowledgeGraph: The loaded knowledge graph
        """
        graph = cls(config)
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Load entities
        for entity_id, entity_data in data.get("entities", {}).items():
            graph.entities[entity_id] = Entity(**entity_data)
            
            # Update indexes
            graph.entity_name_to_id[entity_data["name"].lower()] = entity_id
            graph.entity_type_index[entity_data["type"]].add(entity_id)
            for paper_id in entity_data.get("papers", []):
                graph.entity_paper_index[entity_id].add(paper_id)
        
        # Load relations
        for relation_id, relation_data in data.get("relations", {}).items():
            graph.relations[relation_id] = Relation(**relation_data)
            
            # Update indexes
            graph.relation_type_index[relation_data["type"]].add(relation_id)
            for paper_id in relation_data.get("papers", []):
                graph.paper_relation_index[paper_id].add(relation_id)
            
            # Update entity-relation index
            source_id = relation_data["source_id"]
            target_id = relation_data["target_id"]
            graph.entity_relation_index[source_id].add(relation_id)
            graph.entity_relation_index[target_id].add(relation_id)
        
        # Load papers
        for paper_id, paper_data in data.get("papers", {}).items():
            graph.papers[paper_id] = Paper(**paper_data)
            
            # Update indexes
            for entity_id in paper_data.get("entities", []):
                graph.paper_entity_index[paper_id].add(entity_id)
            
            for relation_id in paper_data.get("relations", []):
                graph.paper_relation_index[paper_id].add(relation_id)
        
        logger.info(f"Loaded knowledge graph from {path} with {len(graph.entities)} entities, {len(graph.relations)} relations, and {len(graph.papers)} papers")
        return graph
    
    def statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge graph.
        
        Returns:
            Dict[str, Any]: Graph statistics
        """
        return {
            "num_entities": len(self.entities),
            "num_relations": len(self.relations),
            "num_papers": len(self.papers),
            "entity_types": {t: len(ids) for t, ids in self.entity_type_index.items()},
            "relation_types": {t: len(ids) for t, ids in self.relation_type_index.items()},
            "top_entities": sorted(
                [(e.id, e.name, len(e.papers)) for e in self.entities.values()],
                key=lambda x: x[2],
                reverse=True
            )[:10]
        }