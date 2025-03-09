"""
Test script for the knowledge graph component.
"""

import os
import pytest
from collections import defaultdict

# Import our components
from src.tools.paper_extraction.knowledge_graph import KnowledgeGraph

def test_knowledge_graph_basic(test_fixtures_dir):
    """Test basic knowledge graph operations."""
    print("Testing Knowledge Graph component...")
    
    # Create a knowledge graph
    storage_dir = os.path.join(test_fixtures_dir, "kg_test")
    os.makedirs(storage_dir, exist_ok=True)
    kg = KnowledgeGraph({"storage_dir": storage_dir})
    
    # Initialize basic data structures for testing
    kg.entities = {}
    kg.relations = {}
    kg.papers = {}
    kg.entity_name_to_id = {}
    kg.entity_type_index = defaultdict(set)
    kg.entity_relation_index = defaultdict(set)
    kg.entity_paper_index = defaultdict(set)
    
    # Get statistics
    stats = kg.statistics()
    print(f"Knowledge Graph initialized with {stats['num_entities']} entities, "
          f"{stats['num_relations']} relations, and {stats['num_papers']} papers")
    
    print("Knowledge Graph test completed successfully!")
    
    # Creating knowledge from paper data manually
    paper_id = "P1"
    paper_knowledge = {
        "title": "Test Paper",
        "authors": ["Author One", "Author Two"],
        "year": "2025",
        "source_paper": {"abstract": "This is a test paper abstract."},
        "entities": [
            {
                "name": "PTEN",
                "type": "Protein",
                "definition": "Tumor suppressor protein",
                "importance_score": 8.5
            },
            {
                "name": "TP53",
                "type": "Gene",
                "definition": "Tumor protein p53",
                "importance_score": 9.0
            }
        ],
        "relations": [
            {
                "source": "PTEN",
                "target": "TP53",
                "relation": "regulates",
                "evidence": "Evidence from experiments",
                "confidence": 0.85
            }
        ]
    }
    
    # Add the knowledge to the knowledge graph
    kg.add_paper_knowledge(paper_id, paper_knowledge)
    
    # Verify everything was added correctly
    assert len(kg.papers) == 1
    assert paper_id in kg.papers
    
    # Test if both entities were added
    assert len(kg.entities) == 2
    
    # Get entity by name
    pten_id = kg.entity_name_to_id.get("pten")
    tp53_id = kg.entity_name_to_id.get("tp53")
    
    assert pten_id is not None
    assert tp53_id is not None
    
    # Check entity properties
    assert kg.entities[pten_id].type == "Protein"
    assert kg.entities[tp53_id].type == "Gene"
    
    # Check the relation between entities
    relation_id = f"{pten_id}|regulates|{tp53_id}"
    assert relation_id in kg.relations
    
    # Verify relation properties
    relation = kg.relations[relation_id]
    assert relation.source_id == pten_id
    assert relation.target_id == tp53_id
    assert relation.type == "regulates"
    assert relation.confidence == 0.85