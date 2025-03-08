"""
Knowledge extraction module for scientific papers.

This module handles semantic understanding and extraction of key concepts,
findings, methodologies, and relationships from scientific papers.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from .pdf_processor import ExtractedPaper

logger = logging.getLogger("co_scientist")

class KnowledgeExtractor:
    """
    Extracts structured knowledge from scientific papers.
    
    This class provides semantic understanding and knowledge extraction
    capabilities, converting unstructured text into structured knowledge
    representations that can be used for synthesis and reasoning.
    """
    
    def __init__(self, config: Dict[str, Any] = None, llm_provider=None):
        """
        Initialize the knowledge extractor.
        
        Args:
            config (Dict[str, Any], optional): Configuration dictionary. Defaults to None.
            llm_provider: LLM provider for semantic extraction. Defaults to None.
        """
        self.config = config or {}
        self.llm_provider = llm_provider
        
        # Configure entity extraction
        self.entity_types = [
            "scientific_concept",
            "methodology",
            "finding",
            "technology",
            "organism",
            "chemical",
            "gene",
            "protein",
            "disease",
            "measurement",
            "theoretical_construct"
        ]
        
        # Configure relation extraction
        self.relation_types = [
            "causes",
            "correlates_with",
            "inhibits",
            "activates",
            "is_part_of",
            "uses",
            "produces",
            "measures",
            "improves",
            "proves",
            "disproves",
            "supports",
            "contradicts"
        ]
        
        # Configure knowledge graph settings
        self.entity_similarity_threshold = self.config.get('entity_similarity_threshold', 0.85)
        
    async def extract_knowledge(self, paper: ExtractedPaper) -> Dict[str, Any]:
        """
        Extract structured knowledge from a scientific paper.
        
        Args:
            paper (ExtractedPaper): The extracted paper content
            
        Returns:
            Dict[str, Any]: Structured knowledge extracted from the paper
        """
        # Initialize knowledge structure
        knowledge = {
            "paper_id": os.path.basename(paper.pdf_path) if paper.pdf_path else "unknown",
            "title": paper.title,
            "authors": paper.authors,
            "year": paper.metadata.get("year", ""),
            "entities": [],
            "relations": [],
            "claims": [],
            "findings": [],
            "methods": [],
            "source_paper": paper.to_dict()
        }
        
        # Extract key entities
        knowledge["entities"] = await self._extract_entities(paper)
        
        # Extract relationships
        knowledge["relations"] = await self._extract_relations(paper, knowledge["entities"])
        
        # Extract scientific claims
        knowledge["claims"] = await self._extract_claims(paper)
        
        # Extract key findings
        knowledge["findings"] = await self._extract_findings(paper)
        
        # Extract methodologies
        knowledge["methods"] = await self._extract_methods(paper)
        
        return knowledge
        
    async def _extract_entities(self, paper: ExtractedPaper) -> List[Dict[str, Any]]:
        """Extract key entities/concepts from the paper."""
        entities = []
        
        if not self.llm_provider:
            logger.warning("No LLM provider available for entity extraction")
            return entities
            
        try:
            # Combine abstract and introduction for context
            context_text = paper.abstract + "\n"
            for section in paper.sections:
                if section.section_type in ['introduction', 'methods', 'results', 'discussion']:
                    context_text += section.content + "\n"
                    if len(context_text) > 2000:  # Limit context length
                        break
            
            # Create prompt for entity extraction
            prompt = f"""
            Extract key scientific entities from this paper. For each entity, identify:
            1. The entity name/term
            2. The entity type (one of: {', '.join(self.entity_types)})
            3. A clear definition/description based on how it's used in the paper
            4. Importance score (1-10) based on centrality to the paper's contribution
            
            Here's the paper text:
            Title: {paper.title}
            
            {context_text}
            
            Return a JSON array with each entity having these properties: "name", "type", "definition", "importance_score". Only include the most important entities (max 15).
            """
            
            # Get response from LLM
            response = await self.llm_provider.generate_text(prompt)
            
            # Parse response to extract entities
            try:
                # Find JSON in response
                import re
                json_match = re.search(r'\[\s*\{.*\}\s*\]', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    entities_data = json.loads(json_str)
                    
                    # Process each entity
                    for entity in entities_data:
                        if "name" in entity and "type" in entity:
                            # Add mention contexts
                            entity["mentions"] = self._find_entity_mentions(paper, entity["name"])
                            entities.append(entity)
                else:
                    logger.warning("Could not find JSON array in LLM response for entity extraction")
            except json.JSONDecodeError:
                logger.error("Failed to parse JSON from LLM response")
                logger.debug(f"Response: {response[:200]}...")
                
        except Exception as e:
            logger.error(f"Error extracting entities: {str(e)}")
            
        return entities
    
    def _find_entity_mentions(self, paper: ExtractedPaper, entity_name: str) -> List[Dict[str, str]]:
        """Find mentions of an entity in the paper text."""
        mentions = []
        
        # Create patterns for exact and approximate matching
        import re
        # Escape for regex
        safe_name = re.escape(entity_name)
        pattern = re.compile(fr'\b{safe_name}\b', re.IGNORECASE)
        
        # Look in abstract
        if paper.abstract:
            matches = pattern.finditer(paper.abstract)
            for match in matches:
                start_idx = max(0, match.start() - 40)
                end_idx = min(len(paper.abstract), match.end() + 40)
                context = paper.abstract[start_idx:end_idx]
                mentions.append({
                    "section": "abstract",
                    "context": context
                })
        
        # Look in sections
        for section in paper.sections:
            if section.content:
                matches = pattern.finditer(section.content)
                for match in matches:
                    start_idx = max(0, match.start() - 40)
                    end_idx = min(len(section.content), match.end() + 40)
                    context = section.content[start_idx:end_idx]
                    mentions.append({
                        "section": section.title,
                        "context": context
                    })
                    
                    # Limit to top 5 mentions
                    if len(mentions) >= 5:
                        return mentions
        
        return mentions
    
    async def _extract_relations(self, paper: ExtractedPaper, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract relationships between entities."""
        relations = []
        
        if not self.llm_provider or not entities:
            logger.warning("No LLM provider or no entities available for relation extraction")
            return relations
            
        try:
            # Get entity names
            entity_names = [entity["name"] for entity in entities]
            
            # Combine information from results and discussion
            context_text = ""
            for section in paper.sections:
                if section.section_type in ['results', 'discussion']:
                    context_text += section.content + "\n"
                    if len(context_text) > 3000:  # Limit context length
                        break
            
            # If context is too small, add introduction
            if len(context_text) < 1000:
                for section in paper.sections:
                    if section.section_type == 'introduction':
                        context_text += section.content + "\n"
                        break
            
            # Create prompt for relation extraction
            prompt = f"""
            Extract relationships between key entities in this paper. For each relationship, identify:
            1. Source entity (from the list provided)
            2. Target entity (from the list provided)
            3. Relationship type (one of: {', '.join(self.relation_types)})
            4. A description or evidence for this relationship from the paper
            5. Confidence score (1-10) based on strength of evidence
            
            Here's the paper text:
            Title: {paper.title}
            
            {context_text}
            
            Key entities to consider: {', '.join(entity_names)}
            
            Return a JSON array with each relationship having these properties: "source", "target", "relation", "evidence", "confidence".
            Only include relationships that are explicitly supported by the paper content. Return at most at most 20 high-confidence relations.
            """
            
            # Get response from LLM
            response = await self.llm_provider.generate_text(prompt)
            
            # Parse response to extract relations
            try:
                # Find JSON in response
                import re
                json_match = re.search(r'\[\s*\{.*\}\s*\]', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    relations_data = json.loads(json_str)
                    
                    # Process each relation, keeping only those between valid entities
                    valid_entity_names = set(entity_names)
                    for relation in relations_data:
                        if ("source" in relation and 
                            "target" in relation and 
                            "relation" in relation and
                            relation["source"] in valid_entity_names and
                            relation["target"] in valid_entity_names):
                            relations.append(relation)
                else:
                    logger.warning("Could not find JSON array in LLM response for relation extraction")
            except json.JSONDecodeError:
                logger.error("Failed to parse JSON from LLM response")
                logger.debug(f"Response: {response[:200]}...")
                
        except Exception as e:
            logger.error(f"Error extracting relations: {str(e)}")
            
        return relations
    
    async def _extract_claims(self, paper: ExtractedPaper) -> List[Dict[str, Any]]:
        """Extract scientific claims from the paper."""
        claims = []
        
        if not self.llm_provider:
            logger.warning("No LLM provider available for claim extraction")
            return claims
            
        try:
            # Combine abstract, introduction, discussion and conclusion for context
            context_text = paper.abstract + "\n"
            for section in paper.sections:
                if section.section_type in ['introduction', 'discussion', 'conclusion']:
                    context_text += section.content + "\n"
                    if len(context_text) > 3000:  # Limit context length
                        break
            
            # Create prompt for claim extraction
            prompt = f"""
            Extract key scientific claims from this paper. For each claim, identify:
            1. The claim statement (a single sentence or short paragraph)
            2. The claim type (one of: observation, causal, correlational, methodological, theoretical)
            3. The evidence or support provided for this claim in the paper
            4. Novelty assessment (1-10) - how novel is this claim compared to prior work
            5. Section where the claim appears
            
            Here's the paper text:
            Title: {paper.title}
            
            {context_text}
            
            Return a JSON array with each claim having these properties: "claim", "type", "evidence", "novelty_score", "section". Focus on the most important claims (max 10).
            """
            
            # Get response from LLM
            response = await self.llm_provider.generate_text(prompt)
            
            # Parse response to extract claims
            try:
                # Find JSON in response
                import re
                json_match = re.search(r'\[\s*\{.*\}\s*\]', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    claims_data = json.loads(json_str)
                    
                    # Process each claim
                    for claim in claims_data:
                        if "claim" in claim and "type" in claim:
                            claims.append(claim)
                else:
                    logger.warning("Could not find JSON array in LLM response for claim extraction")
            except json.JSONDecodeError:
                logger.error("Failed to parse JSON from LLM response")
                logger.debug(f"Response: {response[:200]}...")
                
        except Exception as e:
            logger.error(f"Error extracting claims: {str(e)}")
            
        return claims
    
    async def _extract_findings(self, paper: ExtractedPaper) -> List[Dict[str, Any]]:
        """Extract key findings from the paper."""
        findings = []
        
        if not self.llm_provider:
            logger.warning("No LLM provider available for finding extraction")
            return findings
            
        try:
            # Focus on results and discussion sections
            context_text = paper.abstract + "\n"
            for section in paper.sections:
                if section.section_type in ['results', 'discussion']:
                    context_text += section.content + "\n"
                    if len(context_text) > 3000:  # Limit context length
                        break
            
            # Create prompt for finding extraction
            prompt = f"""
            Extract key scientific findings from this paper. For each finding, identify:
            1. The finding statement (concise description of what was found)
            2. The metrics or measurements supporting this finding (if applicable)
            3. The significance of this finding (why it matters)
            4. Any limitations or uncertainties mentioned about this finding
            5. The section where the finding appears
            
            Here's the paper text:
            Title: {paper.title}
            
            {context_text}
            
            Return a JSON array with each finding having these properties: "finding", "evidence", "significance", "limitations", "section". Focus on the most important findings (max 8).
            """
            
            # Get response from LLM
            response = await self.llm_provider.generate_text(prompt)
            
            # Parse response to extract findings
            try:
                # Find JSON in response
                import re
                json_match = re.search(r'\[\s*\{.*\}\s*\]', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    findings_data = json.loads(json_str)
                    
                    # Process each finding
                    for finding in findings_data:
                        if "finding" in finding:
                            findings.append(finding)
                else:
                    logger.warning("Could not find JSON array in LLM response for finding extraction")
            except json.JSONDecodeError:
                logger.error("Failed to parse JSON from LLM response")
                logger.debug(f"Response: {response[:200]}...")
                
        except Exception as e:
            logger.error(f"Error extracting findings: {str(e)}")
            
        return findings
    
    async def _extract_methods(self, paper: ExtractedPaper) -> List[Dict[str, Any]]:
        """Extract methodologies from the paper."""
        methods = []
        
        if not self.llm_provider:
            logger.warning("No LLM provider available for method extraction")
            return methods
            
        try:
            # Focus on methods section
            context_text = ""
            for section in paper.sections:
                if section.section_type == 'methods':
                    context_text += section.content + "\n"
                    
            # If no methods section found, look for methodology in other sections
            if not context_text:
                for section in paper.sections:
                    if section.section_type in ['introduction', 'results']:
                        context_text += section.content + "\n"
                        if len(context_text) > 3000:  # Limit context length
                            break
            
            # Create prompt for method extraction
            prompt = f"""
            Extract key methodologies from this paper. For each method, identify:
            1. The method name or description
            2. The purpose of this method in the study
            3. Any parameters, settings, or conditions specified
            4. Equipment, materials, or software used (if mentioned)
            5. Any novel aspects of the methodology
            
            Here's the paper text:
            Title: {paper.title}
            
            {context_text}
            
            Return a JSON array with each method having these properties: "method", "purpose", "parameters", "equipment", "novelty". Focus on the most important methods (max 10).
            """
            
            # Get response from LLM
            response = await self.llm_provider.generate_text(prompt)
            
            # Parse response to extract methods
            try:
                # Find JSON in response
                import re
                json_match = re.search(r'\[\s*\{.*\}\s*\]', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    methods_data = json.loads(json_str)
                    
                    # Process each method
                    for method in methods_data:
                        if "method" in method and "purpose" in method:
                            methods.append(method)
                else:
                    logger.warning("Could not find JSON array in LLM response for method extraction")
            except json.JSONDecodeError:
                logger.error("Failed to parse JSON from LLM response")
                logger.debug(f"Response: {response[:200]}...")
                
        except Exception as e:
            logger.error(f"Error extracting methods: {str(e)}")
            
        return methods
    
    def save_knowledge(self, knowledge: Dict[str, Any], output_path: str) -> None:
        """Save the extracted knowledge to a JSON file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(knowledge, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved extracted knowledge to {output_path}")
    
    @staticmethod
    def load_knowledge(input_path: str) -> Dict[str, Any]:
        """Load extracted knowledge from a JSON file."""
        with open(input_path, 'r', encoding='utf-8') as f:
            return json.load(f)