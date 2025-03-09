"""
Test script for the paper extraction and knowledge graph components.
This script demonstrates how to use the paper extraction system to retrieve papers,
extract their content, and build a knowledge graph.
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path
import site

# Add user site-packages to sys.path for access to installed packages
user_site = site.USER_SITE
if user_site not in sys.path:
    sys.path.append(user_site)

# Ensure parent directory is in the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import components
from src.tools.paper_extraction.extraction_manager import PaperExtractionManager
from src.tools.paper_extraction.knowledge_graph import KnowledgeGraph
from src.core.llm_provider import LLMProvider
from src.config.config import load_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_paper_extraction")

async def test_paper_extraction():
    """Test the paper extraction system."""
    # Load config
    config = load_config()
    
    # Initialize LLM provider
    model_config = config.models[config.default_model]
    llm_provider = LLMProvider.create_provider(model_config)
    
    # Create directories
    base_dir = "tests/data/test_fixtures/papers_db"
    os.makedirs(base_dir, exist_ok=True)
    
    kg_dir = "tests/data/test_fixtures/knowledge_graph"
    os.makedirs(kg_dir, exist_ok=True)
    
    # Initialize extraction manager
    extraction_config = {"base_dir": base_dir}
    extraction_manager = PaperExtractionManager(extraction_config, llm_provider=llm_provider)
    
    # Initialize knowledge graph
    graph_config = {"storage_dir": kg_dir}
    kg = KnowledgeGraph(graph_config)
    
    # Test with an ArXiv paper
    arxiv_id = "2302.00924"  # AI for materials discovery paper
    paper_id = f"arxiv_{arxiv_id}"
    arxiv_url = f"https://arxiv.org/abs/{arxiv_id}"
    
    try:
        # Retrieve the paper
        logger.info(f"Retrieving paper with ArXiv URL: {arxiv_url}")
        paper_path = await extraction_manager.retriever.download_pdf(arxiv_url, paper_id)
        
        if paper_path:
            logger.info(f"Paper retrieved and saved to: {paper_path}")
            
            # Process the paper with the real processor
            logger.info("Processing paper using PDF processor...")
            # PyMuPDF should be available from the user site-packages
            paper_data = extraction_manager.processor.process_pdf(paper_path, output_dir=extraction_manager.extraction_dir)
            
            # Save extraction results
            extraction_path = os.path.join(extraction_manager.extraction_dir, f"{paper_id}_extracted.json")
            with open(extraction_path, 'w') as f:
                json.dump(paper_data.to_dict(), f, indent=2)
            
            if paper_data:
                logger.info(f"Paper processed. Extracted {len(paper_data.sections)} sections")
                
                # Extract knowledge from the paper
                logger.info("Extracting knowledge using real extractor...")
                knowledge = await extraction_manager.extractor.extract_knowledge(paper_data)
                
                # Save knowledge results
                knowledge_path = os.path.join(extraction_manager.knowledge_dir, f"{paper_id}_knowledge.json")
                with open(knowledge_path, 'w') as f:
                    json.dump(knowledge, f, indent=2)
                
                if knowledge:
                    logger.info(f"Knowledge extracted. Found {len(knowledge.get('entities', []))} entities and {len(knowledge.get('relations', []))} relations")
                    
                    # Add knowledge to the graph
                    logger.info("Adding knowledge to graph...")
                    kg.add_paper_knowledge(paper_id, knowledge)
                    
                    # Get statistics from the knowledge graph
                    stats = kg.statistics()
                    logger.info(f"Knowledge graph now has {stats['num_entities']} entities, {stats['num_relations']} relations, and {stats['num_papers']} papers")
                    
                    # Save the knowledge graph
                    kg_path = os.path.join(kg_dir, "knowledge_graph.json")
                    kg.save(kg_path)
                    logger.info(f"Knowledge graph saved to: {kg_path}")
                    
                    return True
                else:
                    logger.error("Failed to extract knowledge from the paper")
            else:
                logger.error("Failed to process the paper")
        else:
            logger.error(f"Failed to retrieve paper with ArXiv ID: {arxiv_id}")
    except Exception as e:
        logger.error(f"Error during paper extraction: {str(e)}", exc_info=True)
    
    return False

if __name__ == "__main__":
    success = asyncio.run(test_paper_extraction())
    if success:
        logger.info("Paper extraction test completed successfully!")
    else:
        logger.error("Paper extraction test failed!")