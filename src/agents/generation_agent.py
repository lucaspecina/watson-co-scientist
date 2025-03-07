"""
Generation Agent for generating novel hypotheses, research proposals, and experimental protocols.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple

from .base_agent import BaseAgent
from ..config.config import SystemConfig
from ..core.models import Hypothesis, ResearchGoal, HypothesisSource, Citation, ExperimentalProtocol
from ..tools.web_search import WebSearchTool, ScientificLiteratureSearch

logger = logging.getLogger("co_scientist")

class GenerationAgent(BaseAgent):
    """
    Agent responsible for generating novel research hypotheses, proposals, and experimental protocols.
    
    This agent can:
    1. Generate initial hypotheses based on research goals
    2. Generate literature-grounded hypotheses using scientific search
    3. Generate hypotheses through simulated scientific debates
    4. Generate focused hypotheses on specific topics
    5. Generate experimental protocols to test hypotheses
    6. Generate integrated hypothesis-protocol pairs
    """
    
    def __init__(self, config: SystemConfig):
        """
        Initialize the Generation agent.
        
        Args:
            config (SystemConfig): The system configuration.
        """
        super().__init__("generation", config)
        
        # Initialize web search tools if enabled
        self.web_search = None
        self.literature_search = None
        if config.web_search_enabled:
            # Set provider to "tavily" if available, otherwise use the default
            provider = "tavily" if hasattr(config, "web_search_provider") else "bing"
            api_key = config.web_search_api_key
            
            self.web_search = WebSearchTool(api_key=api_key, provider=provider)
            self.literature_search = ScientificLiteratureSearch(api_key=api_key, provider=provider)
            
    async def generate_initial_hypotheses(self, 
                                     research_goal: ResearchGoal, 
                                     num_hypotheses: int = 3) -> List[Hypothesis]:
        """
        Generate initial hypotheses based on the research goal.
        
        Args:
            research_goal (ResearchGoal): The research goal.
            num_hypotheses (int): The number of hypotheses to generate.
            
        Returns:
            List[Hypothesis]: The generated hypotheses.
        """
        logger.info(f"Generating {num_hypotheses} initial hypotheses for research goal {research_goal.id}")
        
        # Build the prompt
        prompt = f"""
        Generate {num_hypotheses} novel, plausible research hypotheses for the following research goal:
        
        {research_goal.text}
        
        For each hypothesis, provide:
        1. A clear title (one sentence)
        2. A detailed description explaining the hypothesis (1-2 paragraphs)
        3. A brief summary (1-2 sentences)
        4. Key supporting evidence or rationale
        
        Each hypothesis should be novel, grounded in scientific principles, and testable through experimentation.
        
        Format your response as a JSON array of objects with the following structure:
        
        ```json
        [
            {{
                "title": "Hypothesis title",
                "description": "Detailed description...",
                "summary": "Brief summary...",
                "supporting_evidence": ["Evidence 1", "Evidence 2", ...]
            }},
            ...
        ]
        ```
        
        Ensure that each hypothesis addresses the research goal directly and provides a potential explanation or solution.
        """
        
        # Generate hypotheses
        response = await self.generate(prompt)
        
        # Extract the JSON from the response
        try:
            # Find JSON content between backticks or at the start/end of the response
            json_content = response
            if "```json" in response:
                json_content = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_content = response.split("```")[1].split("```")[0].strip()
                
            # Parse the JSON
            hypotheses_data = json.loads(json_content)
            
            # Convert to Hypothesis objects
            hypotheses = []
            for data in hypotheses_data:
                hypothesis = Hypothesis(
                    title=data["title"],
                    description=data["description"],
                    summary=data["summary"],
                    supporting_evidence=data["supporting_evidence"],
                    creator="generation",
                    source=HypothesisSource.SYSTEM,
                    metadata={"research_goal_id": research_goal.id}
                )
                hypotheses.append(hypothesis)
                
            logger.info(f"Generated {len(hypotheses)} hypotheses")
            return hypotheses
            
        except Exception as e:
            logger.error(f"Error parsing hypotheses from response: {e}")
            logger.debug(f"Response: {response}")
            return []
    
    async def generate_hypotheses_with_literature(self, 
                                             research_goal: ResearchGoal, 
                                             num_hypotheses: int = 3) -> List[Hypothesis]:
        """
        Generate hypotheses based on the research goal with literature search.
        
        Args:
            research_goal (ResearchGoal): The research goal.
            num_hypotheses (int): The number of hypotheses to generate.
            
        Returns:
            List[Hypothesis]: The generated hypotheses.
        """
        logger.info(f"Generating {num_hypotheses} hypotheses with literature for research goal {research_goal.id}")
        
        if not self.literature_search:
            logger.warning("Scientific literature search is disabled. Falling back to generating hypotheses without literature.")
            return await self.generate_initial_hypotheses(research_goal, num_hypotheses)
            
        # Perform a scientific literature search
        query = f"latest research {research_goal.text}"
        search_result = await self.literature_search.search_with_citations(query, max_results=5)
        
        # Extract results and citations
        search_results = search_result.get("results", [])
        citations = search_result.get("citations", [])
        
        if not search_results:
            logger.warning("No scientific literature found. Falling back to generating hypotheses without literature.")
            return await self.generate_initial_hypotheses(research_goal, num_hypotheses)
            
        # Build the prompt with search results
        literature_context = "\n\n".join([
            f"Source {i+1}: {result.get('title', 'Untitled')}\n"
            f"URL: {result.get('url', 'No URL')}\n"
            f"Summary: {result.get('snippet', 'No snippet available')}"
            for i, result in enumerate(search_results)
        ])
        
        # Create a citation reference guide for the model
        citation_guide = "\n".join([
            f"[{i+1}] {citation.get('title', 'Untitled')} - {citation.get('url', 'No URL')}"
            for i, citation in enumerate(citations)
        ])
        
        prompt = f"""
        Below is a research goal followed by summaries of relevant scientific literature. Based on this information, generate {num_hypotheses} novel, plausible research hypotheses that build upon the existing literature.

        Research Goal:
        {research_goal.text}
        
        Relevant Scientific Literature:
        {literature_context}
        
        Available Citations:
        {citation_guide}
        
        For each hypothesis, provide:
        1. A clear title (one sentence)
        2. A detailed description explaining the hypothesis (1-2 paragraphs)
        3. A brief summary (1-2 sentences)
        4. Key supporting evidence or rationale from the literature
        5. References to the scientific literature (use the citation numbers from the Available Citations section)
        
        Each hypothesis should be novel, grounded in the scientific literature, and testable through experimentation.
        
        Format your response as a JSON array of objects with the following structure:
        
        ```json
        [
            {{
                "title": "Hypothesis title",
                "description": "Detailed description...",
                "summary": "Brief summary...",
                "supporting_evidence": ["Evidence 1", "Evidence 2", ...],
                "citation_ids": [1, 2, 3]
            }},
            ...
        ]
        ```
        
        Ensure that each hypothesis addresses the research goal directly and builds upon the existing literature in a novel way.
        Include specific citation references (by number) to support your hypotheses.
        """
        
        # Generate hypotheses
        response = await self.generate(prompt)
        
        # Extract the JSON from the response
        try:
            # Find JSON content between backticks or at the start/end of the response
            json_content = response
            if "```json" in response:
                json_content = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_content = response.split("```")[1].split("```")[0].strip()
                
            # Parse the JSON
            hypotheses_data = json.loads(json_content)
            
            # Convert to Hypothesis objects with citations
            hypotheses = []
            for data in hypotheses_data:
                # Create citation objects from the citation IDs
                hypothesis_citations = []
                
                # Check if citation_ids exists and is a list
                citation_ids = data.get("citation_ids", [])
                if isinstance(citation_ids, list):
                    for cid in citation_ids:
                        # Convert citation ID to 0-based index
                        idx = int(cid) - 1
                        if 0 <= idx < len(citations):
                            citation_info = citations[idx]
                            citation = Citation(
                                title=citation_info.get("title", "Unknown"),
                                url=citation_info.get("url", ""),
                                authors=[],  # We don't have author information easily available
                                snippet=citation_info.get("snippet", ""),
                                source="literature_search",
                                publication_date=citation_info.get("publication_date", ""),
                                metadata={
                                    "research_goal_id": research_goal.id,
                                    "citation_index": cid
                                }
                            )
                            hypothesis_citations.append(citation)
                
                # Create the hypothesis with citations
                hypothesis = Hypothesis(
                    title=data["title"],
                    description=data["description"],
                    summary=data["summary"],
                    supporting_evidence=data["supporting_evidence"],
                    citations=hypothesis_citations,  # Add the citations
                    creator="generation_with_literature",
                    source=HypothesisSource.SYSTEM,
                    literature_grounded=True,
                    metadata={
                        "research_goal_id": research_goal.id,
                        "literature_search_query": query
                    }
                )
                hypotheses.append(hypothesis)
                
            logger.info(f"Generated {len(hypotheses)} hypotheses with literature and {len(citations)} citations")
            return hypotheses
            
        except Exception as e:
            logger.error(f"Error parsing hypotheses from response: {e}")
            logger.debug(f"Response: {response}")
            return []
    
    async def generate_hypotheses_debate(self,
                                    research_goal: ResearchGoal,
                                    num_hypotheses: int = 1,
                                    with_literature: bool = True) -> List[Hypothesis]:
        """
        Generate hypotheses using a simulated scientific debate.
        
        Args:
            research_goal (ResearchGoal): The research goal.
            num_hypotheses (int): The number of hypotheses to generate.
            with_literature (bool): Whether to ground the debate in scientific literature.
            
        Returns:
            List[Hypothesis]: The generated hypotheses.
        """
        logger.info(f"Generating {num_hypotheses} hypotheses through debate for research goal {research_goal.id}")
        
        # If literature grounding is requested but not available, log a warning
        literature_context = ""
        citations = []
        
        if with_literature and self.literature_search:
            # Perform a scientific literature search
            query = f"latest scientific research {research_goal.text}"
            search_result = await self.literature_search.search_with_citations(query, max_results=3)
            
            # Extract results and citations
            search_results = search_result.get("results", [])
            citations = search_result.get("citations", [])
            
            if search_results:
                # Build the literature context
                literature_context = "\n\n".join([
                    f"Source {i+1}: {result.get('title', 'Untitled')}\n"
                    f"URL: {result.get('url', 'No URL')}\n"
                    f"Summary: {result.get('snippet', 'No snippet available')}"
                    for i, result in enumerate(search_results)
                ])
                
                # Create a citation reference guide
                citation_guide = "\n".join([
                    f"[{i+1}] {citation.get('title', 'Untitled')} - {citation.get('url', 'No URL')}"
                    for i, citation in enumerate(citations)
                ])
                
                literature_context = f"""
                Relevant Scientific Literature:
                {literature_context}
                
                Available Citations:
                {citation_guide}
                """
        
        # Build the prompt for a scientific debate
        prompt = f"""
        You will simulate a scientific debate between multiple experts to generate a novel research hypothesis for the following research goal:
        
        {research_goal.text}
        
        {literature_context}
        
        The debate will proceed in three rounds:
        
        Round 1: Each expert proposes an initial hypothesis related to the research goal.
        Round 2: Experts critique each other's hypotheses, pointing out weaknesses and suggesting improvements.
        Round 3: Experts collaborate to synthesize a final, improved hypothesis that addresses the critiques.
        
        After the debate, summarize the final hypothesis with:
        1. A clear title (one sentence)
        2. A detailed description explaining the hypothesis (1-2 paragraphs)
        3. A brief summary (1-2 sentences)
        4. Key supporting evidence or rationale
        5. References to the scientific literature (use the citation numbers if provided)
        
        Format the final output as a JSON object with the following structure:
        
        ```json
        {{
            "title": "Hypothesis title",
            "description": "Detailed description...",
            "summary": "Brief summary...",
            "supporting_evidence": ["Evidence 1", "Evidence 2", ...],
            "citation_ids": [1, 2, 3],
            "debate_transcript": "Full transcript of the debate..."
        }}
        ```
        
        Ensure that the final hypothesis is novel, scientifically plausible, and directly addresses the research goal.
        """
        
        # Generate hypothesis through debate
        response = await self.generate(prompt)
        
        # Extract the JSON from the response
        try:
            # Find JSON content between backticks or at the start/end of the response
            json_content = response
            if "```json" in response:
                json_content = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_content = response.split("```")[1].split("```")[0].strip()
                
            # Parse the JSON
            data = json.loads(json_content)
            
            # Process citations if available
            hypothesis_citations = []
            citation_ids = data.get("citation_ids", [])
            
            if with_literature and isinstance(citation_ids, list) and len(citations) > 0:
                for cid in citation_ids:
                    # Convert citation ID to 0-based index
                    idx = int(cid) - 1
                    if 0 <= idx < len(citations):
                        citation_info = citations[idx]
                        citation = Citation(
                            title=citation_info.get("title", "Unknown"),
                            url=citation_info.get("url", ""),
                            authors=[],  # We don't have author information easily available
                            snippet=citation_info.get("snippet", ""),
                            source="debate_literature_search",
                            publication_date=citation_info.get("publication_date", ""),
                            metadata={
                                "research_goal_id": research_goal.id,
                                "citation_index": cid
                            }
                        )
                        hypothesis_citations.append(citation)
            
            # Convert to Hypothesis object
            hypothesis = Hypothesis(
                title=data["title"],
                description=data["description"],
                summary=data["summary"],
                supporting_evidence=data["supporting_evidence"],
                citations=hypothesis_citations,
                creator="generation_debate",
                source=HypothesisSource.SYSTEM,
                literature_grounded=with_literature and len(hypothesis_citations) > 0,
                metadata={
                    "research_goal_id": research_goal.id,
                    "debate_transcript": data.get("debate_transcript", "")
                }
            )
            
            logger.info(f"Generated hypothesis through debate: {hypothesis.title}")
            return [hypothesis]
            
        except Exception as e:
            logger.error(f"Error parsing hypothesis from debate response: {e}")
            logger.debug(f"Response: {response}")
            return []
            
    async def generate_grounded_hypothesis(self, research_goal: ResearchGoal, topic: str) -> Optional[Hypothesis]:
        """
        Generate a single hypothesis that is specifically focused on a given topic and grounded in literature.
        
        Args:
            research_goal (ResearchGoal): The research goal.
            topic (str): The specific topic to focus on.
            
        Returns:
            Optional[Hypothesis]: The generated hypothesis or None if generation fails.
        """
        logger.info(f"Generating focused hypothesis on topic '{topic}' for research goal {research_goal.id}")
        
        if not self.literature_search:
            logger.warning("Scientific literature search is disabled. Cannot generate grounded hypothesis.")
            return None
            
        # Perform a targeted literature search on the specific topic
        query = f"{topic} research {research_goal.text}"
        search_result = await self.literature_search.search_with_citations(query, max_results=5)
        
        # Extract results and citations
        search_results = search_result.get("results", [])
        citations = search_result.get("citations", [])
        
        if not search_results:
            logger.warning(f"No scientific literature found for topic '{topic}'.")
            return None
            
        # Build the prompt with search results
        literature_context = "\n\n".join([
            f"Source {i+1}: {result.get('title', 'Untitled')}\n"
            f"URL: {result.get('url', 'No URL')}\n"
            f"Summary: {result.get('snippet', 'No snippet available')}"
            for i, result in enumerate(search_results)
        ])
        
        # Create a citation reference guide for the model
        citation_guide = "\n".join([
            f"[{i+1}] {citation.get('title', 'Untitled')} - {citation.get('url', 'No URL')}"
            for i, citation in enumerate(citations)
        ])
        
        prompt = f"""
        Below is a research goal and a specific topic to focus on, followed by summaries of relevant scientific literature. 
        Based on this information, generate ONE novel, plausible research hypothesis that builds upon the existing literature.

        Research Goal:
        {research_goal.text}
        
        Focus Topic:
        {topic}
        
        Relevant Scientific Literature:
        {literature_context}
        
        Available Citations:
        {citation_guide}
        
        Please provide:
        1. A clear title (one sentence)
        2. A detailed description explaining the hypothesis (1-2 paragraphs)
        3. A brief summary (1-2 sentences)
        4. Key supporting evidence or rationale from the literature
        5. References to the scientific literature (use the citation numbers from the Available Citations section)
        
        The hypothesis should be novel, grounded in the scientific literature, highly relevant to the focus topic, and testable through experimentation.
        
        Format your response as a JSON object with the following structure:
        
        ```json
        {{
            "title": "Hypothesis title",
            "description": "Detailed description...",
            "summary": "Brief summary...",
            "supporting_evidence": ["Evidence 1", "Evidence 2", ...],
            "citation_ids": [1, 2, 3],
            "relevance_explanation": "Explanation of how this relates to the focus topic..."
        }}
        ```
        """
        
        # Generate the hypothesis
        response = await self.generate(prompt)
        
        # Extract the JSON from the response
        try:
            # Find JSON content between backticks or at the start/end of the response
            json_content = response
            if "```json" in response:
                json_content = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_content = response.split("```")[1].split("```")[0].strip()
                
            # Parse the JSON
            data = json.loads(json_content)
            
            # Process citations
            hypothesis_citations = []
            citation_ids = data.get("citation_ids", [])
            
            if isinstance(citation_ids, list):
                for cid in citation_ids:
                    # Convert citation ID to 0-based index
                    idx = int(cid) - 1
                    if 0 <= idx < len(citations):
                        citation_info = citations[idx]
                        citation = Citation(
                            title=citation_info.get("title", "Unknown"),
                            url=citation_info.get("url", ""),
                            authors=[],  # We don't have author information easily available
                            snippet=citation_info.get("snippet", ""),
                            source="focused_literature_search",
                            publication_date=citation_info.get("publication_date", ""),
                            metadata={
                                "research_goal_id": research_goal.id,
                                "citation_index": cid
                            }
                        )
                        hypothesis_citations.append(citation)
            
            # Create the hypothesis with citations
            hypothesis = Hypothesis(
                title=data["title"],
                description=data["description"],
                summary=data["summary"],
                supporting_evidence=data["supporting_evidence"],
                citations=hypothesis_citations,
                creator="generation_focused",
                source=HypothesisSource.SYSTEM,
                literature_grounded=True,
                metadata={
                    "research_goal_id": research_goal.id,
                    "focus_topic": topic,
                    "literature_search_query": query,
                    "relevance_explanation": data.get("relevance_explanation", "")
                }
            )
            
            logger.info(f"Generated focused hypothesis on topic '{topic}': {hypothesis.title}")
            return hypothesis
            
        except Exception as e:
            logger.error(f"Error parsing focused hypothesis from response: {e}")
            logger.debug(f"Response: {response}")
            return None
            
    async def generate_experimental_protocol(self, 
                                      hypothesis: Hypothesis, 
                                      research_goal: ResearchGoal) -> Optional[ExperimentalProtocol]:
        """
        Generate an experimental protocol to test a given hypothesis.
        
        Args:
            hypothesis (Hypothesis): The hypothesis to test.
            research_goal (ResearchGoal): The research goal.
            
        Returns:
            Optional[ExperimentalProtocol]: The generated experimental protocol or None if generation fails.
        """
        logger.info(f"Generating experimental protocol for hypothesis {hypothesis.id}: {hypothesis.title}")
        
        # Look for citations if literature search is enabled
        literature_context = ""
        if self.literature_search:
            # Search for experimental methods related to the hypothesis
            query = f"{hypothesis.title} experimental methods protocols techniques"
            search_result = await self.literature_search.search_with_citations(query, max_results=3)
            
            # Extract results and citations
            search_results = search_result.get("results", [])
            
            if search_results:
                # Format the literature context
                literature_sources = "\n\n".join([
                    f"Source {i+1}: {result.get('title', 'Untitled')}\n"
                    f"URL: {result.get('url', 'No URL')}\n"
                    f"Summary: {result.get('snippet', 'No snippet available')}"
                    for i, result in enumerate(search_results)
                ])
                
                literature_context = f"""
                ## Relevant Experimental Methods from Literature
                Use these sources to inform your thinking about appropriate experimental methods:
                
                {literature_sources}
                """
                
        # Build the prompt
        prompt = f"""
        You are tasked with generating a detailed experimental protocol to test the following scientific hypothesis.
        
        Research Goal:
        {research_goal.text}
        
        Hypothesis to Test:
        Title: {hypothesis.title}
        Summary: {hypothesis.summary}
        Description: {hypothesis.description}
        Supporting Evidence: {', '.join(hypothesis.supporting_evidence)}
        
        {literature_context}
        
        Please design a comprehensive experimental protocol that would allow scientists to test this hypothesis effectively.
        The protocol should be detailed, practical, and scientifically rigorous.
        
        Include the following components:
        1. A clear title for the protocol
        2. A detailed description of the overall approach
        3. Step-by-step experimental procedures
        4. Materials required (reagents, equipment, samples, etc.)
        5. Equipment needed
        6. Expected results if the hypothesis is correct
        7. Potential limitations or challenges of the protocol
        8. Controls to ensure validity of results
        9. Statistical analysis approach
        
        Format your response as a JSON object with the following structure:
        
        ```json
        {{
            "title": "Title of the experimental protocol",
            "description": "Detailed description of the protocol approach...",
            "steps": [
                "Step 1: ...",
                "Step 2: ...",
                ...
            ],
            "materials": [
                "Material 1",
                "Material 2",
                ...
            ],
            "equipment": [
                "Equipment 1",
                "Equipment 2",
                ...
            ],
            "expected_results": "Description of expected results if hypothesis is correct...",
            "limitations": [
                "Limitation 1",
                "Limitation 2",
                ...
            ],
            "controls": [
                "Control 1: ...",
                "Control 2: ...",
                ...
            ],
            "analysis_approach": "Description of statistical or analytical approach..."
        }}
        ```
        
        The protocol should be technically feasible with current technology and scientific methods.
        Make sure the protocol directly tests the key aspects of the hypothesis.
        """
        
        # Generate the protocol
        response = await self.generate(prompt)
        
        # Extract the JSON from the response
        try:
            # Find JSON content between backticks or at the start/end of the response
            json_content = response
            if "```json" in response:
                json_content = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_content = response.split("```")[1].split("```")[0].strip()
                
            # Parse the JSON
            data = json.loads(json_content)
            
            # Format the steps to include controls and analysis if they exist
            steps = data["steps"]
            
            # Add controls to steps if they exist
            if "controls" in data and data["controls"]:
                steps.append("## Controls")
                steps.extend(data["controls"])
                
            # Add analysis approach to steps if it exists
            if "analysis_approach" in data and data["analysis_approach"]:
                steps.append("## Analysis Approach")
                steps.append(data["analysis_approach"])
            
            # Create the experimental protocol
            protocol = ExperimentalProtocol(
                hypothesis_id=hypothesis.id,
                title=data["title"],
                description=data["description"],
                steps=steps,
                materials=data["materials"],
                equipment=data["equipment"],
                expected_results=data["expected_results"],
                limitations=data.get("limitations", []),
                creator="generation",
                metadata={
                    "research_goal_id": hypothesis.metadata.get("research_goal_id"),
                    "controls": data.get("controls", []),
                    "analysis_approach": data.get("analysis_approach", "")
                }
            )
            
            logger.info(f"Generated experimental protocol for hypothesis {hypothesis.id}: {protocol.title}")
            return protocol
            
        except Exception as e:
            logger.error(f"Error parsing experimental protocol from response: {e}")
            logger.debug(f"Response: {response}")
            return None
            
    async def generate_hypothesis_with_protocol(self, 
                                         research_goal: ResearchGoal) -> Tuple[Optional[Hypothesis], Optional[ExperimentalProtocol]]:
        """
        Generate a hypothesis and corresponding experimental protocol in a single step.
        
        Args:
            research_goal (ResearchGoal): The research goal.
            
        Returns:
            Tuple[Optional[Hypothesis], Optional[ExperimentalProtocol]]: The generated hypothesis and protocol pair.
        """
        logger.info(f"Generating hypothesis with integrated protocol for research goal {research_goal.id}")
        
        # First generate a literature-grounded hypothesis if possible
        if self.literature_search:
            hypothesis = await self.generate_hypotheses_with_literature(research_goal, num_hypotheses=1)
            if hypothesis and len(hypothesis) > 0:
                # Now generate the experimental protocol
                protocol = await self.generate_experimental_protocol(hypothesis[0], research_goal)
                return hypothesis[0], protocol
        
        # Fallback to non-literature approach
        hypothesis = await self.generate_initial_hypotheses(research_goal, num_hypotheses=1)
        if hypothesis and len(hypothesis) > 0:
            # Generate the experimental protocol
            protocol = await self.generate_experimental_protocol(hypothesis[0], research_goal)
            return hypothesis[0], protocol
            
        return None, None