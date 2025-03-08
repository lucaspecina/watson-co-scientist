#!/usr/bin/env python3
"""
Simple test script for the knowledge graph component.
"""

import os
import sys
import site
from collections import defaultdict

# Add user site-packages to path (where pip installs packages for the user)
user_site = site.USER_SITE
if user_site not in sys.path:
    sys.path.append(user_site)

# Now import our components
from src.tools.paper_extraction.knowledge_graph import KnowledgeGraph

def main():
    """Test the knowledge graph."""
    print("Testing Knowledge Graph component...")
    
    # Create a knowledge graph
    kg = KnowledgeGraph({"storage_dir": "data/kg_test"})
    
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
    
if __name__ == "__main__":
    main()