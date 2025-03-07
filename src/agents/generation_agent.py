"""
Generation Agent for generating novel hypotheses and research proposals.
"""

import json
import logging
from typing import Dict, List, Any, Optional

from .base_agent import BaseAgent
from ..config.config import SystemConfig
from ..core.models import Hypothesis, ResearchGoal
from ..tools.web_search import WebSearchTool

logger = logging.getLogger("co_scientist")

class GenerationAgent(BaseAgent):
    """
    Agent responsible for generating novel research hypotheses and proposals.
    """
    
    def __init__(self, config: SystemConfig):
        """
        Initialize the Generation agent.
        
        Args:
            config (SystemConfig): The system configuration.
        """
        super().__init__("generation", config)
        
        # Initialize web search tool if enabled
        self.web_search = None
        if config.web_search_enabled:
            self.web_search = WebSearchTool(config.web_search_api_key)
            
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
        
        if not self.web_search:
            logger.warning("Web search is disabled. Falling back to generating hypotheses without literature.")
            return await self.generate_initial_hypotheses(research_goal, num_hypotheses)
            
        # Perform a literature search
        query = f"latest research {research_goal.text}"
        search_results = await self.web_search.search(query, count=5)
        
        if not search_results:
            logger.warning("No search results found. Falling back to generating hypotheses without literature.")
            return await self.generate_initial_hypotheses(research_goal, num_hypotheses)
            
        # Build the prompt with search results
        literature_context = "\n\n".join([
            f"Title: {result['title']}\nURL: {result['url']}\nSummary: {result['snippet']}"
            for result in search_results
        ])
        
        prompt = f"""
        Below is a research goal followed by summaries of relevant scientific literature. Based on this information, generate {num_hypotheses} novel, plausible research hypotheses that build upon the existing literature.

        Research Goal:
        {research_goal.text}
        
        Relevant Literature:
        {literature_context}
        
        For each hypothesis, provide:
        1. A clear title (one sentence)
        2. A detailed description explaining the hypothesis (1-2 paragraphs)
        3. A brief summary (1-2 sentences)
        4. Key supporting evidence or rationale from the literature
        
        Each hypothesis should be novel, grounded in the scientific literature, and testable through experimentation.
        
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
        
        Ensure that each hypothesis addresses the research goal directly and builds upon the existing literature in a novel way.
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
                    creator="generation_with_literature",
                    metadata={"research_goal_id": research_goal.id}
                )
                hypotheses.append(hypothesis)
                
            logger.info(f"Generated {len(hypotheses)} hypotheses with literature")
            return hypotheses
            
        except Exception as e:
            logger.error(f"Error parsing hypotheses from response: {e}")
            logger.debug(f"Response: {response}")
            return []
    
    async def generate_hypotheses_debate(self,
                                    research_goal: ResearchGoal,
                                    num_hypotheses: int = 1) -> List[Hypothesis]:
        """
        Generate hypotheses using a simulated scientific debate.
        
        Args:
            research_goal (ResearchGoal): The research goal.
            num_hypotheses (int): The number of hypotheses to generate.
            
        Returns:
            List[Hypothesis]: The generated hypotheses.
        """
        logger.info(f"Generating {num_hypotheses} hypotheses through debate for research goal {research_goal.id}")
        
        # Build the prompt for a scientific debate
        prompt = f"""
        You will simulate a scientific debate between multiple experts to generate a novel research hypothesis for the following research goal:
        
        {research_goal.text}
        
        The debate will proceed in three rounds:
        
        Round 1: Each expert proposes an initial hypothesis related to the research goal.
        Round 2: Experts critique each other's hypotheses, pointing out weaknesses and suggesting improvements.
        Round 3: Experts collaborate to synthesize a final, improved hypothesis that addresses the critiques.
        
        After the debate, summarize the final hypothesis with:
        1. A clear title (one sentence)
        2. A detailed description explaining the hypothesis (1-2 paragraphs)
        3. A brief summary (1-2 sentences)
        4. Key supporting evidence or rationale
        
        Format the final output as a JSON object with the following structure:
        
        ```json
        {{
            "title": "Hypothesis title",
            "description": "Detailed description...",
            "summary": "Brief summary...",
            "supporting_evidence": ["Evidence 1", "Evidence 2", ...],
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
            
            # Convert to Hypothesis object
            hypothesis = Hypothesis(
                title=data["title"],
                description=data["description"],
                summary=data["summary"],
                supporting_evidence=data["supporting_evidence"],
                creator="generation_debate",
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