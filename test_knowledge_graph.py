#!/usr/bin/env python3
"""
Test script for the Knowledge Graph component of the Paper Knowledge Extraction System.

This script demonstrates the capabilities of the Knowledge Graph by loading papers,
extracting knowledge, and performing graph-based operations.
"""

import os
import asyncio
import argparse
import json
from pprint import pprint
from datetime import datetime
from dotenv import load_dotenv

# Add src to the path so we can import from it
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# Import necessary components
from src.core.llm_provider import get_llm_provider
from src.tools.paper_extraction.extraction_manager import PaperExtractionManager
from src.tools.paper_extraction.knowledge_graph import KnowledgeGraph
from src.utils.logger import setup_logger

# Load .env variables
load_dotenv()

async def process_papers_to_graph(papers_dir, output_path=None, llm_provider=None):
    """Process papers into a knowledge graph."""
    print(f"\n===== Processing papers from {papers_dir} into knowledge graph =====")
    
    # Configure extraction manager
    extraction_config = {
        'base_dir': 'data/papers_db'
    }
    
    # Configure knowledge graph
    graph_config = {
        'storage_dir': 'data/knowledge_graph'
    }
    
    # Create extraction manager and knowledge graph
    manager = PaperExtractionManager(extraction_config, llm_provider=llm_provider)
    graph = KnowledgeGraph(graph_config)
    
    # Get list of papers from extraction manager
    papers = manager.list_papers()
    print(f"Found {len(papers)} processed papers to add to graph")
    
    # Process each paper
    for paper in papers:
        paper_id = paper['id']
        
        # Check if paper has knowledge
        if not paper['has_knowledge']:
            if llm_provider:
                print(f"Paper {paper_id} has no knowledge. Extracting knowledge...")
                
                # Extract knowledge
                if paper['pdf_path']:
                    knowledge = await manager.extract_from_pdf(paper['pdf_path'], paper_id)
                else:
                    print(f"Paper {paper_id} has no PDF path. Skipping...")
                    continue
            else:
                print(f"Paper {paper_id} has no knowledge and no LLM provider. Skipping...")
                continue
        else:
            # Load knowledge
            knowledge = manager.get_knowledge(paper_id)
            
            if not knowledge:
                print(f"Failed to load knowledge for paper {paper_id}. Skipping...")
                continue
        
        # Add knowledge to graph
        print(f"Adding paper {paper_id} to knowledge graph...")
        graph.add_paper_knowledge(paper_id, knowledge)
    
    # Save graph
    if output_path:
        graph.save(output_path)
    else:
        graph.save()
        
    # Output statistics
    stats = graph.statistics()
    print("\nKnowledge Graph Statistics:")
    print(f"  Entities: {stats['num_entities']}")
    print(f"  Relations: {stats['num_relations']}")
    print(f"  Papers: {stats['num_papers']}")
    
    if stats["entity_types"]:
        print("\nEntity Types:")
        for entity_type, count in stats["entity_types"].items():
            print(f"  {entity_type}: {count}")
    
    if stats["relation_types"]:
        print("\nRelation Types:")
        for relation_type, count in stats["relation_types"].items():
            print(f"  {relation_type}: {count}")
    
    if stats["top_entities"]:
        print("\nTop Entities (by paper count):")
        for i, (entity_id, name, count) in enumerate(stats["top_entities"], 1):
            print(f"  {i}. {name} ({count} papers)")
    
    return graph

async def explore_graph(graph_path=None):
    """Explore a knowledge graph."""
    print("\n===== Exploring Knowledge Graph =====")
    
    # Configure knowledge graph
    graph_config = {
        'storage_dir': 'data/knowledge_graph'
    }
    
    # Load graph
    if graph_path:
        graph = KnowledgeGraph.load(graph_path, graph_config)
    else:
        default_path = os.path.join(graph_config['storage_dir'], "knowledge_graph.json")
        if os.path.exists(default_path):
            graph = KnowledgeGraph.load(default_path, graph_config)
        else:
            print(f"Error: No knowledge graph found at {default_path}")
            return None
    
    # Output statistics
    stats = graph.statistics()
    print("\nKnowledge Graph Statistics:")
    print(f"  Entities: {stats['num_entities']}")
    print(f"  Relations: {stats['num_relations']}")
    print(f"  Papers: {stats['num_papers']}")
    
    # Interactive exploration
    print("\nExploration Options:")
    print("1. View all entity types")
    print("2. View all relation types")
    print("3. Search for entity by name")
    print("4. View entity neighborhood")
    print("5. Find path between entities")
    print("6. Find common entities across papers")
    print("7. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-7): ")
        
        if choice == "1":
            # View all entity types
            entity_types = stats["entity_types"]
            print("\nEntity Types:")
            for i, (entity_type, count) in enumerate(sorted(entity_types.items(), key=lambda x: x[1], reverse=True), 1):
                print(f"  {i}. {entity_type}: {count}")
                
            # Ask if user wants to see entities of a specific type
            type_choice = input("\nEnter entity type to view entities (or press Enter to skip): ")
            if type_choice.strip():
                entities = graph.find_entities_by_type(type_choice)
                if entities:
                    print(f"\nEntities of type '{type_choice}':")
                    for i, entity in enumerate(sorted(entities, key=lambda e: e.importance, reverse=True), 1):
                        print(f"  {i}. {entity.name} (Importance: {entity.importance:.1f}, Papers: {len(entity.papers)})")
                        if i <= 3:  # Show definition for top 3
                            print(f"     Definition: {entity.definition[:100]}...")
                else:
                    print(f"No entities found of type '{type_choice}'")
        
        elif choice == "2":
            # View all relation types
            relation_types = stats["relation_types"]
            print("\nRelation Types:")
            for i, (relation_type, count) in enumerate(sorted(relation_types.items(), key=lambda x: x[1], reverse=True), 1):
                print(f"  {i}. {relation_type}: {count}")
                
            # Ask if user wants to see relations of a specific type
            type_choice = input("\nEnter relation type to view relations (or press Enter to skip): ")
            if type_choice.strip():
                relations = graph.find_relations_by_type(type_choice)
                if relations:
                    print(f"\nRelations of type '{type_choice}':")
                    for i, relation in enumerate(sorted(relations, key=lambda r: r.confidence, reverse=True), 1):
                        source = graph.get_entity(relation.source_id)
                        target = graph.get_entity(relation.target_id)
                        source_name = source.name if source else "Unknown"
                        target_name = target.name if target else "Unknown"
                        print(f"  {i}. {source_name} --{relation.type}--> {target_name} (Confidence: {relation.confidence:.1f})")
                        if i <= 3:  # Show evidence for top 3
                            print(f"     Evidence: {relation.evidence[:100]}...")
                else:
                    print(f"No relations found of type '{type_choice}'")
        
        elif choice == "3":
            # Search for entity by name
            entity_name = input("\nEnter entity name to search for: ")
            partial = input("Allow partial matches? (y/n): ").lower() == 'y'
            
            if entity_name.strip():
                entities = graph.find_entities_by_name(entity_name, partial_match=partial)
                if entities:
                    print(f"\nEntities matching '{entity_name}':")
                    for i, entity in enumerate(entities, 1):
                        print(f"  {i}. {entity.name} (Type: {entity.type}, Importance: {entity.importance:.1f})")
                        print(f"     Definition: {entity.definition[:100]}...")
                        print(f"     Papers: {len(entity.papers)}")
                        
                        # Show papers mentioning this entity
                        papers = graph.get_entity_papers(entity.id)
                        if papers:
                            print(f"     Mentioned in:")
                            for j, paper in enumerate(papers[:3], 1):
                                print(f"       {j}. {paper.title}")
                                
                        # Show relations involving this entity
                        relations = graph.get_entity_relations(entity.id)
                        if relations:
                            print(f"     Relations:")
                            for j, relation in enumerate(relations[:5], 1):
                                other_id = relation.target_id if relation.source_id == entity.id else relation.source_id
                                other = graph.get_entity(other_id)
                                other_name = other.name if other else "Unknown"
                                print(f"       {j}. {relation.type} {other_name}")
                else:
                    print(f"No entities found matching '{entity_name}'")
        
        elif choice == "4":
            # View entity neighborhood
            entity_name = input("\nEnter entity name to view neighborhood: ")
            depth = int(input("Enter neighborhood depth (1-3): ") or "1")
            depth = max(1, min(3, depth))  # Limit to 1-3
            
            if entity_name.strip():
                entities = graph.find_entities_by_name(entity_name)
                if entities:
                    entity = entities[0]
                    print(f"\nViewing neighborhood for entity '{entity.name}':")
                    
                    # Get neighborhood
                    neighborhood = graph.get_entity_neighborhood(entity.id, max_depth=depth)
                    
                    # Print entities
                    print(f"\nEntities in neighborhood ({len(neighborhood['entities'])}):")
                    for i, (entity_id, entity_data) in enumerate(neighborhood["entities"].items(), 1):
                        print(f"  {i}. {entity_data['name']} (Type: {entity_data['type']})")
                    
                    # Print relations
                    print(f"\nRelations in neighborhood ({len(neighborhood['relations'])}):")
                    for i, (relation_id, relation_data) in enumerate(neighborhood["relations"].items(), 1):
                        source = graph.get_entity(relation_data['source_id'])
                        target = graph.get_entity(relation_data['target_id'])
                        source_name = source.name if source else "Unknown"
                        target_name = target.name if target else "Unknown"
                        print(f"  {i}. {source_name} --{relation_data['type']}--> {target_name}")
                else:
                    print(f"No entities found matching '{entity_name}'")
        
        elif choice == "5":
            # Find path between entities
            source_name = input("\nEnter source entity name: ")
            target_name = input("Enter target entity name: ")
            max_depth = int(input("Enter maximum path depth (1-5): ") or "3")
            max_depth = max(1, min(5, max_depth))  # Limit to 1-5
            
            if source_name.strip() and target_name.strip():
                source_entities = graph.find_entities_by_name(source_name)
                target_entities = graph.find_entities_by_name(target_name)
                
                if source_entities and target_entities:
                    source = source_entities[0]
                    target = target_entities[0]
                    
                    print(f"\nFinding paths from '{source.name}' to '{target.name}':")
                    
                    # Find paths
                    paths = graph.find_path(source.id, target.id, max_depth=max_depth)
                    
                    if paths:
                        print(f"\nFound {len(paths)} paths:")
                        for i, path in enumerate(paths, 1):
                            print(f"  Path {i}:")
                            for j, (source_id, relation_type, target_id) in enumerate(path, 1):
                                source_entity = graph.get_entity(source_id)
                                target_entity = graph.get_entity(target_id)
                                source_name = source_entity.name if source_entity else "Unknown"
                                target_name = target_entity.name if target_entity else "Unknown"
                                print(f"    {j}. {source_name} --{relation_type}--> {target_name}")
                    else:
                        print(f"No paths found from '{source.name}' to '{target.name}' with max depth {max_depth}")
                else:
                    if not source_entities:
                        print(f"No entities found matching source '{source_name}'")
                    if not target_entities:
                        print(f"No entities found matching target '{target_name}'")
        
        elif choice == "6":
            # Find common entities across papers
            print("\nPapers in graph:")
            for i, (paper_id, paper) in enumerate(graph.papers.items(), 1):
                print(f"  {i}. {paper.title}")
                
            paper_ids_input = input("\nEnter paper IDs (comma-separated) to find common entities: ")
            paper_ids = [p.strip() for p in paper_ids_input.split(",") if p.strip()]
            
            if paper_ids:
                common_entities = graph.find_common_entities(paper_ids)
                if common_entities:
                    print(f"\nFound {len(common_entities)} common entities:")
                    for i, entity in enumerate(sorted(common_entities, key=lambda e: e.importance, reverse=True), 1):
                        print(f"  {i}. {entity.name} (Type: {entity.type}, Importance: {entity.importance:.1f})")
                        print(f"     Definition: {entity.definition[:100]}...")
                else:
                    print("No common entities found across the specified papers")
            else:
                print("No valid paper IDs provided")
        
        elif choice == "7":
            # Exit
            print("\nExiting exploration...")
            break
        
        else:
            print("\nInvalid choice. Please enter a number from 1 to 7.")
    
    return graph

async def main():
    """Main function."""
    # Set up logging
    logger = setup_logger()
    
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Test Knowledge Graph')
    parser.add_argument('--process', action='store_true', help='Process papers into knowledge graph')
    parser.add_argument('--explore', action='store_true', help='Explore knowledge graph')
    parser.add_argument('--papers-dir', help='Directory containing processed papers')
    parser.add_argument('--graph-path', help='Path to knowledge graph file')
    parser.add_argument('--output', help='Path to output knowledge graph file')
    parser.add_argument('--no-llm', action='store_true', help='Disable LLM-based extraction')
    
    args = parser.parse_args()
    
    # Initialize LLM provider if requested
    llm_provider = None
    if not args.no_llm:
        try:
            llm_provider = get_llm_provider()
            print("LLM provider initialized successfully")
        except Exception as e:
            print(f"Failed to initialize LLM provider: {e}")
            print("Running without LLM-based extraction")
    
    # Process papers into graph
    if args.process:
        papers_dir = args.papers_dir or 'data/papers_db'
        graph = await process_papers_to_graph(papers_dir, args.output, llm_provider)
    
    # Explore graph
    if args.explore:
        await explore_graph(args.graph_path)
    
    # If no action specified, show usage
    if not args.process and not args.explore:
        parser.print_help()

if __name__ == "__main__":
    asyncio.run(main())