#!/usr/bin/env python
"""
Demonstration script for the paper extraction and knowledge graph components.

This script demonstrates how to use the paper extraction system to retrieve papers,
extract their content, build a knowledge graph, and visualize the connections.
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path
import argparse

# Ensure parent directory is in the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import components
from src.tools.paper_extraction.extraction_manager import PaperExtractionManager
from src.tools.paper_extraction.knowledge_graph import KnowledgeGraph
from src.core.llm_provider import LLMProvider
from src.config.config import load_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("demo_paper_extraction")

async def demo_paper_extraction(paper_url=None, topic=None, visualize=False):
    """
    Demonstrate the paper extraction system by extracting knowledge from scientific papers
    and building a knowledge graph.
    
    Args:
        paper_url (str, optional): URL of a specific paper to extract. Defaults to None.
        topic (str, optional): Scientific topic to search for papers. Defaults to None.
        visualize (bool, optional): Whether to generate a visualization. Defaults to False.
    """
    # Load config
    config = load_config()
    
    # Initialize LLM provider
    model_config = config.models[config.default_model]
    llm_provider = LLMProvider.create_provider(model_config)
    
    # Create directories
    base_dir = "data/papers_db"
    os.makedirs(base_dir, exist_ok=True)
    
    kg_dir = "data/knowledge_graph"
    os.makedirs(kg_dir, exist_ok=True)
    
    # Initialize extraction manager
    extraction_config = {"base_dir": base_dir}
    extraction_manager = PaperExtractionManager(extraction_config, llm_provider=llm_provider)
    
    # Initialize knowledge graph
    graph_config = {"storage_dir": kg_dir}
    kg = KnowledgeGraph(graph_config)
    
    # Load existing knowledge graph if available
    graph_path = os.path.join(kg_dir, "knowledge_graph.json")
    if os.path.exists(graph_path):
        try:
            kg = KnowledgeGraph.load(graph_path, graph_config)
            logger.info(f"Loaded existing knowledge graph with {kg.statistics()['num_entities']} entities, "
                       f"{kg.statistics()['num_relations']} relations, and {kg.statistics()['num_papers']} papers")
        except Exception as e:
            logger.error(f"Failed to load knowledge graph: {str(e)}")
            kg = KnowledgeGraph(graph_config)
    
    # Process specific paper if URL provided
    if paper_url:
        await process_paper(extraction_manager, kg, paper_url)
    
    # Search for papers by topic if provided
    elif topic:
        await search_and_process_papers(extraction_manager, kg, topic)
    
    # If neither URL nor topic provided, use a default topic
    else:
        default_topic = "antimicrobial resistance horizontal gene transfer"
        logger.info(f"No paper URL or topic provided. Using default topic: '{default_topic}'")
        await search_and_process_papers(extraction_manager, kg, default_topic)
    
    # Get statistics from the knowledge graph
    stats = kg.statistics()
    logger.info(f"Knowledge graph now has {stats['num_entities']} entities, "
               f"{stats['num_relations']} relations, and {stats['num_papers']} papers")
    
    # Save the knowledge graph
    kg.save(graph_path)
    logger.info(f"Knowledge graph saved to: {graph_path}")
    
    # Explore the knowledge graph
    await explore_knowledge_graph(kg)
    
    # Visualize the knowledge graph if requested
    if visualize:
        try:
            visualize_graph(kg, os.path.join(kg_dir, "knowledge_graph_viz.html"))
        except Exception as e:
            logger.error(f"Error visualizing knowledge graph: {str(e)}")

async def process_paper(extraction_manager, kg, paper_url):
    """Process a single paper from URL and add to knowledge graph."""
    logger.info(f"Processing paper from URL: {paper_url}")
    
    try:
        # Extract paper ID from URL
        paper_id = None
        
        # Handle arXiv URLs
        if "arxiv.org" in paper_url:
            if "abs/" in paper_url:
                arxiv_id = paper_url.split("abs/")[1].split()[0].strip()
                paper_id = f"arxiv_{arxiv_id}"
            elif "pdf/" in paper_url:
                arxiv_id = paper_url.split("pdf/")[1].split(".pdf")[0].strip()
                paper_id = f"arxiv_{arxiv_id}"
        
        # Handle PubMed URLs
        elif "pubmed.ncbi.nlm.nih.gov" in paper_url:
            pubmed_id = paper_url.split("/")[-1].strip()
            paper_id = f"pubmed_{pubmed_id}"
            
        # Use generic ID if no specific pattern recognized
        if not paper_id:
            import hashlib
            paper_id = f"paper_{hashlib.md5(paper_url.encode()).hexdigest()[:8]}"
        
        # Retrieve the paper
        logger.info(f"Retrieving paper with URL: {paper_url}")
        paper_path = await extraction_manager.retriever.download_pdf(paper_url, paper_id)
        
        if paper_path:
            logger.info(f"Paper retrieved and saved to: {paper_path}")
            
            # Process the paper
            logger.info("Processing paper using PDF processor...")
            paper_data = extraction_manager.processor.process_pdf(paper_path, 
                                                                 output_dir=extraction_manager.extraction_dir)
            
            # Save extraction results
            extraction_path = os.path.join(extraction_manager.extraction_dir, f"{paper_id}_extracted.json")
            with open(extraction_path, 'w') as f:
                json.dump(paper_data.to_dict(), f, indent=2)
            
            if paper_data:
                logger.info(f"Paper processed. Extracted {len(paper_data.sections)} sections")
                
                # Extract knowledge from the paper
                logger.info("Extracting knowledge using LLM...")
                knowledge = await extraction_manager.extractor.extract_knowledge(paper_data)
                
                # Save knowledge results
                knowledge_path = os.path.join(extraction_manager.knowledge_dir, f"{paper_id}_knowledge.json")
                with open(knowledge_path, 'w') as f:
                    json.dump(knowledge, f, indent=2)
                
                if knowledge:
                    logger.info(f"Knowledge extracted. Found {len(knowledge.get('entities', []))} entities and "
                               f"{len(knowledge.get('relations', []))} relations")
                    
                    # Add knowledge to the graph
                    logger.info("Adding knowledge to graph...")
                    kg.add_paper_knowledge(paper_id, knowledge)
                    
                    return True
                else:
                    logger.error("Failed to extract knowledge from the paper")
            else:
                logger.error("Failed to process the paper")
        else:
            logger.error(f"Failed to retrieve paper with URL: {paper_url}")
    except Exception as e:
        logger.error(f"Error processing paper: {str(e)}")
    
    return False

async def search_and_process_papers(extraction_manager, kg, topic, max_papers=3):
    """Search for papers on a topic and process them."""
    logger.info(f"Searching for papers on topic: '{topic}'")
    
    try:
        # Use the web search tool directly for paper search since extraction_manager doesn't have search_papers
        from src.tools.web_search import WebSearchTool
        from src.config.config import load_config
        
        config = load_config()
        search_tool = WebSearchTool(config.web_search_api_key)
        
        # Construct search query for scientific papers
        search_query = f"{topic} scientific paper pdf site:arxiv.org OR site:pubmed.ncbi.nlm.nih.gov"
        logger.info(f"Performing web search with query: '{search_query}'")
        
        # Search for papers
        search_results = await search_tool.search(search_query, count=max_papers, search_type="scientific")
        
        if not search_results:
            logger.warning(f"No papers found for topic: '{topic}'")
            return False
        
        # Filter results to include only PDF links or scientific repositories
        paper_urls = []
        for result in search_results:
            url = result.get('url', '')
            if ('arxiv.org' in url or 'pubmed' in url or 'ncbi.nlm.nih.gov' in url or 
                url.endswith('.pdf') or '/pdf/' in url):
                paper_urls.append(url)
        
        if not paper_urls:
            logger.warning(f"No suitable paper URLs found for topic: '{topic}'")
            paper_urls = [result['url'] for result in search_results[:max_papers]]
        
        logger.info(f"Found {len(paper_urls)} papers on topic: '{topic}'")
        
        # Process each paper
        successful_papers = 0
        for paper_url in paper_urls:
            success = await process_paper(extraction_manager, kg, paper_url)
            if success:
                successful_papers += 1
        
        logger.info(f"Successfully processed {successful_papers} out of {len(paper_urls)} papers")
        return successful_papers > 0
        
    except Exception as e:
        logger.error(f"Error searching and processing papers: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

async def explore_knowledge_graph(kg):
    """Explore and display interesting insights from the knowledge graph."""
    logger.info("Exploring knowledge graph for insights...")
    
    try:
        # Check if graph is populated
        if not kg.entities:
            logger.info("Knowledge graph is empty. No insights to explore.")
            return
        
        # Check available entities
        logger.info(f"Knowledge graph contains {len(kg.entities)} entities")
        
        # Get entities with their names for display
        entity_names = {}
        for entity_id, entity in kg.entities.items():
            if isinstance(entity, dict):
                entity_names[entity_id] = entity.get('name', entity_id)
            else:
                # Object access
                entity_names[entity_id] = getattr(entity, 'name', entity_id)
                
        # Get relation counts by entity
        entity_relations = {}
        for entity_id in kg.entities:
            entity_relations[entity_id] = 0
            
        # Try to inspect relations - could be in different formats
        paper_entities = {}
        try:
            # First assume relations is an object with items() method
            for rel_id, rel_data in kg.relations.items():
                try:
                    # Dictionary access
                    if isinstance(rel_data, dict):
                        source_id = rel_data.get('source_id')
                        target_id = rel_data.get('target_id')
                        paper_id = rel_data.get('paper_id')
                    else:
                        # Object access
                        source_id = getattr(rel_data, 'source_id', None)
                        target_id = getattr(rel_data, 'target_id', None)
                        paper_id = getattr(rel_data, 'paper_id', None)
                        
                    if source_id in entity_relations:
                        entity_relations[source_id] += 1
                    if target_id in entity_relations:
                        entity_relations[target_id] += 1
                        
                    if paper_id:
                        if paper_id not in paper_entities:
                            paper_entities[paper_id] = set()
                        paper_entities[paper_id].add(source_id)
                        paper_entities[paper_id].add(target_id)
                except Exception as e:
                    pass
        except Exception as e:
            # If relations is not dictionary-like, try list-like
            try:
                for rel in kg.relations:
                    try:
                        if isinstance(rel, dict):
                            source_id = rel.get('source_id')
                            target_id = rel.get('target_id')
                            paper_id = rel.get('paper_id')
                        else:
                            source_id = getattr(rel, 'source_id', None)
                            target_id = getattr(rel, 'target_id', None)
                            paper_id = getattr(rel, 'paper_id', None)
                            
                        if source_id in entity_relations:
                            entity_relations[source_id] += 1
                        if target_id in entity_relations:
                            entity_relations[target_id] += 1
                            
                        if paper_id:
                            if paper_id not in paper_entities:
                                paper_entities[paper_id] = set()
                            paper_entities[paper_id].add(source_id)
                            paper_entities[paper_id].add(target_id)
                    except Exception:
                        pass
            except Exception as e:
                logger.warning(f"Could not analyze relations: {e}")
                
        # Display top entities by connection count
        top_entities = sorted(entity_relations.items(), key=lambda x: x[1], reverse=True)[:5]
        if top_entities:
            logger.info("Most connected entities in the knowledge graph:")
            for i, (entity_id, count) in enumerate(top_entities):
                logger.info(f"  {i+1}. {entity_names.get(entity_id, entity_id)} (Connections: {count})")
                
        # Show papers and their entities
        if hasattr(kg, 'papers') and kg.papers:
            logger.info("Papers in knowledge graph:")
            paper_count = 0
            for paper_id, paper_data in kg.papers.items():
                paper_count += 1
                if isinstance(paper_data, dict):
                    title = paper_data.get('title', paper_id)
                else:
                    title = getattr(paper_data, 'title', paper_id)
                
                logger.info(f"  {paper_count}. {title}")
                
                # Show entities for this paper
                if paper_id in paper_entities:
                    entity_ids = list(paper_entities[paper_id])[:5]  # Limit to 5 entities
                    logger.info(f"     Related entities: {', '.join([entity_names.get(e, e) for e in entity_ids])}")
            
        # Show relation types if available
        if hasattr(kg, 'get_relation_types'):
            relation_types = kg.get_relation_types()
            if relation_types:
                logger.info("Relation types in the knowledge graph:")
                for i, rel_type in enumerate(relation_types[:5]):  # Limit to top 5
                    logger.info(f"  {i+1}. {rel_type}")
                    
    except Exception as e:
        logger.error(f"Error exploring knowledge graph: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

def visualize_graph(kg, output_path):
    """Generate a visualization of the knowledge graph."""
    try:
        import networkx as nx
        from pyvis.network import Network
        
        logger.info("Generating knowledge graph visualization...")
        
        # Create NetworkX graph
        G = nx.DiGraph()
        
        # Add entity nodes
        for entity_id, entity in kg.entities.items():
            if isinstance(entity, dict):
                G.add_node(entity_id, 
                          label=entity.get('name', entity_id), 
                          title=entity.get('definition', ''), 
                          group=entity.get('type', 'entity'))
            else:
                # For object entities
                G.add_node(entity_id, 
                          label=getattr(entity, 'name', entity_id), 
                          title=getattr(entity, 'definition', ''), 
                          group=getattr(entity, 'type', 'entity'))
        
        # Add relations as edges
        try:
            # First try dictionary access for relations
            for rel_id, relation in kg.relations.items():
                try:
                    if isinstance(relation, dict):
                        source_id = relation.get('source_id')
                        target_id = relation.get('target_id')
                        rel_type = relation.get('type')
                    else:
                        source_id = getattr(relation, 'source_id', None)
                        target_id = getattr(relation, 'target_id', None)
                        rel_type = getattr(relation, 'type', None)
                        
                    if source_id and target_id:
                        G.add_edge(source_id, target_id, title=rel_type, label=rel_type)
                except Exception:
                    pass
        except AttributeError:
            # Try list access for relations
            try:
                for relation in kg.relations:
                    try:
                        if isinstance(relation, dict):
                            source_id = relation.get('source_id')
                            target_id = relation.get('target_id')
                            rel_type = relation.get('type')
                        else:
                            source_id = getattr(relation, 'source_id', None)
                            target_id = getattr(relation, 'target_id', None)
                            rel_type = getattr(relation, 'type', None)
                            
                        if source_id and target_id:
                            G.add_edge(source_id, target_id, title=rel_type, label=rel_type)
                    except Exception:
                        pass
            except Exception:
                logger.error("Could not process relations for visualization")
        
        # Create Pyvis network
        net = Network(height="800px", width="100%", notebook=False, directed=True)
        
        # Use physics for better layout
        net.barnes_hut(gravity=-5000, central_gravity=0.3, spring_length=150)
        
        # Use NetworkX graph as input
        net.from_nx(G)
        
        # Save visualization
        net.save_graph(output_path)
        logger.info(f"Visualization saved to: {output_path}")
        
    except ImportError:
        logger.error("Visualization requires networkx and pyvis packages. Install with: pip install networkx pyvis")
    except Exception as e:
        logger.error(f"Error generating visualization: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demonstrate paper extraction and knowledge graph building")
    parser.add_argument("--url", help="URL of a paper to extract and analyze")
    parser.add_argument("--topic", help="Topic to search for papers")
    parser.add_argument("--visualize", action="store_true", help="Generate a visualization of the knowledge graph")
    
    args = parser.parse_args()
    
    asyncio.run(demo_paper_extraction(args.url, args.topic, args.visualize))