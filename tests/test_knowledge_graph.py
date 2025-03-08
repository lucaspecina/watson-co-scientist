"""
Test the Scientific Knowledge Graph component of the Paper Extraction System.

This script tests the KnowledgeGraph's ability to:
1. Create and manage a scientific knowledge graph
2. Add entities, relations, and papers to the graph
3. Query the graph for entities, relationships, and papers
4. Find paths between entities in the graph
5. Perform network analysis on the graph
"""

import os
import json
import tempfile
import pytest
import networkx as nx
from unittest.mock import patch, MagicMock

from src.tools.paper_extraction.knowledge_graph import KnowledgeGraph


class TestKnowledgeGraph:
    """Test the KnowledgeGraph component."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Use a temporary directory for tests
        self.temp_dir = tempfile.mkdtemp()
        self.config = {"storage_dir": self.temp_dir}
        self.graph = KnowledgeGraph(self.config)
        
        # Add some test data
        self.setup_test_data()
        
    def setup_test_data(self):
        """Set up test data for the knowledge graph."""
        # Add entities
        self.graph.add_entity({
            "id": "protein:p53",
            "name": "p53",
            "type": "protein",
            "description": "Tumor suppressor protein"
        })
        self.graph.add_entity({
            "id": "gene:MDM2",
            "name": "MDM2",
            "type": "gene",
            "description": "E3 ubiquitin-protein ligase"
        })
        self.graph.add_entity({
            "id": "pathway:apoptosis",
            "name": "Apoptosis",
            "type": "pathway",
            "description": "Programmed cell death"
        })
        self.graph.add_entity({
            "id": "disease:cancer",
            "name": "Cancer",
            "type": "disease",
            "description": "Disease characterized by abnormal cell growth"
        })
        
        # Add papers
        self.graph.add_paper({
            "id": "paper1",
            "title": "p53 and cancer",
            "authors": ["Author 1", "Author 2"],
            "year": "2020",
            "journal": "Nature",
            "abstract": "This paper discusses p53's role in cancer.",
            "url": "https://example.com/paper1"
        })
        self.graph.add_paper({
            "id": "paper2",
            "title": "MDM2 regulation",
            "authors": ["Author 3"],
            "year": "2021",
            "journal": "Science",
            "abstract": "This paper examines how MDM2 is regulated.",
            "url": "https://example.com/paper2"
        })
        
        # Add relations
        self.graph.add_relation({
            "source": "protein:p53",
            "target": "gene:MDM2",
            "relation_type": "regulates",
            "paper_id": "paper1"
        })
        self.graph.add_relation({
            "source": "gene:MDM2",
            "target": "pathway:apoptosis",
            "relation_type": "inhibits",
            "paper_id": "paper2"
        })
        self.graph.add_relation({
            "source": "protein:p53",
            "target": "pathway:apoptosis",
            "relation_type": "activates",
            "paper_id": "paper1"
        })
        self.graph.add_relation({
            "source": "pathway:apoptosis",
            "target": "disease:cancer",
            "relation_type": "prevents",
            "paper_id": "paper1"
        })
    
    def test_initialization(self):
        """Test that the KnowledgeGraph initializes correctly."""
        graph = KnowledgeGraph(self.config)
        assert graph is not None
        assert hasattr(graph, 'entities')
        assert hasattr(graph, 'relations')
        assert hasattr(graph, 'papers')
        assert hasattr(graph, 'config')
        
    def test_add_entity(self):
        """Test adding an entity to the graph."""
        # Add a new entity
        self.graph.add_entity({
            "id": "chemical:cisplatin",
            "name": "Cisplatin",
            "type": "chemical",
            "description": "Chemotherapy drug"
        })
        
        # Verify it was added
        assert "chemical:cisplatin" in self.graph.entities
        assert self.graph.entities["chemical:cisplatin"]["name"] == "Cisplatin"
        assert self.graph.entities["chemical:cisplatin"]["type"] == "chemical"
        
    def test_add_paper(self):
        """Test adding a paper to the graph."""
        # Add a new paper
        self.graph.add_paper({
            "id": "paper3",
            "title": "Apoptosis mechanisms",
            "authors": ["Author 4", "Author 5"],
            "year": "2022",
            "journal": "Cell",
            "abstract": "This paper discusses mechanisms of apoptosis.",
            "url": "https://example.com/paper3"
        })
        
        # Verify it was added
        assert "paper3" in self.graph.papers
        assert self.graph.papers["paper3"]["title"] == "Apoptosis mechanisms"
        assert "Author 4" in self.graph.papers["paper3"]["authors"]
        
    def test_add_relation(self):
        """Test adding a relation to the graph."""
        # Add a new relation
        self.graph.add_relation({
            "source": "protein:p53",
            "target": "disease:cancer",
            "relation_type": "suppresses",
            "paper_id": "paper1"
        })
        
        # Verify it was added
        # Find the relation we just added
        found = False
        for relation in self.graph.relations:
            if (relation["source"] == "protein:p53" and 
                relation["target"] == "disease:cancer" and 
                relation["relation_type"] == "suppresses"):
                found = True
                break
        
        assert found, "Relation was not added correctly"
        
    def test_get_entity(self):
        """Test getting an entity by ID."""
        entity = self.graph.get_entity("protein:p53")
        assert entity is not None
        assert entity["name"] == "p53"
        assert entity["type"] == "protein"
        
        # Test getting a non-existent entity
        assert self.graph.get_entity("not_an_entity") is None
        
    def test_get_paper(self):
        """Test getting a paper by ID."""
        paper = self.graph.get_paper("paper1")
        assert paper is not None
        assert paper["title"] == "p53 and cancer"
        assert "Author 1" in paper["authors"]
        
        # Test getting a non-existent paper
        assert self.graph.get_paper("not_a_paper") is None
        
    def test_find_entities(self):
        """Test finding entities by search term."""
        # Search by exact name
        entities = self.graph.find_entities("p53")
        assert len(entities) >= 1
        assert "protein:p53" in entities
        
        # Search by partial name
        entities = self.graph.find_entities("apop")
        assert len(entities) >= 1
        assert "pathway:apoptosis" in entities
        
        # Search by description
        entities = self.graph.find_entities("suppress")
        assert len(entities) >= 1
        assert "protein:p53" in entities
        
        # Search with no results
        entities = self.graph.find_entities("not_a_term")
        assert len(entities) == 0
        
    def test_find_papers(self):
        """Test finding papers by search term."""
        # Search by title
        papers = self.graph.find_papers("cancer")
        assert len(papers) >= 1
        assert "paper1" in papers
        
        # Search by author
        papers = self.graph.find_papers("Author 1")
        assert len(papers) >= 1
        assert "paper1" in papers
        
        # Search by abstract
        papers = self.graph.find_papers("regulate")
        assert len(papers) >= 1
        assert "paper2" in papers
        
        # Search with no results
        papers = self.graph.find_papers("not_a_term")
        assert len(papers) == 0
        
    def test_get_entity_relations(self):
        """Test getting relations for an entity."""
        # Get outgoing relations
        relations = self.graph.get_entity_relations("protein:p53")
        assert len(relations) >= 2  # Should have at least 2 relations (MDM2 and apoptosis)
        
        # Verify one of the relations
        found_mdm2 = False
        for relation in relations:
            if relation["target_entity"] == "gene:MDM2":
                found_mdm2 = True
                assert relation["relation_type"] == "regulates"
                break
        
        assert found_mdm2, "Relation to MDM2 not found"
        
        # Test with non-existent entity
        relations = self.graph.get_entity_relations("not_an_entity")
        assert len(relations) == 0
        
    def test_find_path(self):
        """Test finding a path between entities."""
        # Direct path
        path = self.graph.find_path("protein:p53", "gene:MDM2")
        assert path is not None
        assert len(path) == 2
        assert path[0] == "protein:p53"
        assert path[1] == "gene:MDM2"
        
        # Path with intermediate entity
        path = self.graph.find_path("protein:p53", "disease:cancer")
        assert path is not None
        assert len(path) > 2  # Either via apoptosis or direct
        assert path[0] == "protein:p53"
        assert path[-1] == "disease:cancer"
        
        # No path
        path = self.graph.find_path("protein:p53", "non_existent")
        assert path is None
        
    def test_find_common_entities(self):
        """Test finding common entities between papers."""
        # Add entities to papers
        self.graph.add_entity_to_paper("protein:p53", "paper1")
        self.graph.add_entity_to_paper("protein:p53", "paper2")
        self.graph.add_entity_to_paper("gene:MDM2", "paper2")
        
        # Find common entities
        common = self.graph.find_common_entities(["paper1", "paper2"])
        assert "protein:p53" in common
        assert "gene:MDM2" not in common  # Only in paper2
        
        # Test with no common entities
        self.graph.add_entity_to_paper("pathway:apoptosis", "paper1")
        common = self.graph.find_common_entities(["paper1", "paper3"])  # paper3 has no entities
        assert len(common) == 0
        
    def test_find_papers_with_entity(self):
        """Test finding papers containing an entity."""
        # Add entities to papers
        self.graph.add_entity_to_paper("protein:p53", "paper1")
        self.graph.add_entity_to_paper("protein:p53", "paper2")
        
        # Find papers with entity
        papers = self.graph.find_papers_with_entity("protein:p53")
        assert len(papers) == 2
        assert "paper1" in papers
        assert "paper2" in papers
        
        # Test with no papers
        papers = self.graph.find_papers_with_entity("non_existent")
        assert len(papers) == 0
        
    def test_save_and_load(self):
        """Test saving and loading the graph."""
        # Save the graph
        save_path = os.path.join(self.temp_dir, "test_graph.json")
        self.graph.save(save_path)
        
        # Create a new graph and load
        new_graph = KnowledgeGraph(self.config)
        new_graph.load(save_path)
        
        # Verify the loaded graph matches the original
        assert len(new_graph.entities) == len(self.graph.entities)
        assert len(new_graph.relations) == len(self.graph.relations)
        assert len(new_graph.papers) == len(self.graph.papers)
        
        # Check a specific entity
        assert "protein:p53" in new_graph.entities
        assert new_graph.entities["protein:p53"]["name"] == "p53"
        
    def test_calculate_entity_centrality(self):
        """Test calculating centrality of entities in the graph."""
        # Calculate centrality
        centrality = self.graph.calculate_entity_centrality()
        
        # Verify results
        assert "protein:p53" in centrality
        assert "pathway:apoptosis" in centrality
        
        # p53 should have high centrality (connected to multiple entities)
        assert centrality["protein:p53"] > 0
        
    def test_generate_entity_neighbors(self):
        """Test generating a neighborhood of entities around a focal entity."""
        # Generate neighborhood
        neighborhood = self.graph.generate_entity_neighbors("protein:p53", max_distance=2)
        
        # Verify results
        assert "protein:p53" in neighborhood  # The focal entity
        assert "gene:MDM2" in neighborhood    # Direct connection
        assert "pathway:apoptosis" in neighborhood  # Direct or 2-step connection
        
        # Test with invalid entity
        neighborhood = self.graph.generate_entity_neighbors("non_existent")
        assert len(neighborhood) == 0
        
    def test_get_statistics(self):
        """Test getting statistics about the graph."""
        stats = self.graph.get_statistics()
        
        # Verify statistics
        assert "entity_count" in stats
        assert stats["entity_count"] == len(self.graph.entities)
        
        assert "relation_count" in stats
        assert stats["relation_count"] == len(self.graph.relations)
        
        assert "paper_count" in stats
        assert stats["paper_count"] == len(self.graph.papers)
        
        # Verify entity type distribution
        assert "entity_types" in stats
        assert "protein" in stats["entity_types"]
        assert "gene" in stats["entity_types"]
        assert "pathway" in stats["entity_types"]
        assert "disease" in stats["entity_types"]


if __name__ == "__main__":
    # Run the tests directly when the script is executed
    pytest.main(["-xvs", __file__])