"""
Evolution Agent for improving existing hypotheses.
"""

import json
import logging
import random
from typing import Dict, List, Any, Optional, Tuple

from .base_agent import BaseAgent
from ..config.config import SystemConfig
from ..core.models import Hypothesis, ResearchGoal, Review
from ..tools.web_search import WebSearchTool

logger = logging.getLogger("co_scientist")

class EvolutionAgent(BaseAgent):
    """
    Agent responsible for improving existing hypotheses.
    """
    
    def __init__(self, config: SystemConfig):
        """
        Initialize the Evolution agent.
        
        Args:
            config (SystemConfig): The system configuration.
        """
        super().__init__("evolution", config)
        
        # Initialize web search tool if enabled
        self.web_search = None
        if config.web_search_enabled:
            self.web_search = WebSearchTool(config.web_search_api_key)
    
    async def improve_hypothesis(self, 
                            hypothesis: Hypothesis, 
                            research_goal: ResearchGoal,
                            reviews: List[Review] = None) -> Hypothesis:
        """
        Improve an existing hypothesis based on reviews and research goal.
        
        Args:
            hypothesis (Hypothesis): The hypothesis to improve.
            research_goal (ResearchGoal): The research goal.
            reviews (List[Review]): List of reviews for the hypothesis.
            
        Returns:
            Hypothesis: The improved hypothesis.
        """
        logger.info(f"Improving hypothesis {hypothesis.id}: {hypothesis.title}")
        
        # Prepare review context if provided
        review_context = ""
        if reviews:
            review_text = "\n\n".join([
                f"Review {i+1}:\n{review.text}"
                for i, review in enumerate(reviews)
            ])
            
            review_context = f"""
            Previous Reviews:
            {review_text}
            """
        
        # Build the prompt
        prompt = f"""
        You are improving an existing scientific hypothesis based on the research goal and previous reviews.
        
        Research Goal:
        {research_goal.text}
        
        Original Hypothesis:
        Title: {hypothesis.title}
        Summary: {hypothesis.summary}
        Description: {hypothesis.description}
        Supporting Evidence: {', '.join(hypothesis.supporting_evidence)}
        
        {review_context}
        
        Your task is to create an improved version of this hypothesis that:
        1. Addresses any weaknesses or critiques from the reviews
        2. Maintains or enhances the strengths identified in reviews
        3. Makes the hypothesis more precise, testable, and aligned with the research goal
        4. Incorporates additional relevant scientific principles or evidence
        5. Improves the clarity and coherence of the hypothesis
        
        Do not simply tweak the hypothesis; make substantive improvements while preserving the core idea.
        
        Format your response as a JSON object with the following structure:
        
        ```json
        {{
            "title": "New concise title for the hypothesis",
            "summary": "Brief summary (1-2 sentences)",
            "description": "Detailed description of the improved hypothesis (1-2 paragraphs)",
            "supporting_evidence": ["Evidence 1", "Evidence 2", ...],
            "improvements_made": ["Improvement 1", "Improvement 2", ...],
            "rationale": "Explanation of why these improvements address the critiques and strengthen the hypothesis"
        }}
        ```
        """
        
        # Generate improved hypothesis
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
            
            # Create the improved hypothesis
            improved = Hypothesis(
                title=data["title"],
                description=data["description"],
                summary=data["summary"],
                supporting_evidence=data["supporting_evidence"],
                creator="evolution",
                parent_hypotheses=[hypothesis.id],
                metadata={
                    "research_goal_id": research_goal.id,
                    "improvements_made": data.get("improvements_made", []),
                    "improvement_rationale": data.get("rationale", "")
                }
            )
            
            logger.info(f"Created improved hypothesis: {improved.title}")
            return improved
            
        except Exception as e:
            logger.error(f"Error parsing improved hypothesis from response: {e}")
            logger.debug(f"Response: {response}")
            
            # Return the original hypothesis in case of error
            return hypothesis
    
    async def combine_hypotheses(self, 
                            hypotheses: List[Hypothesis], 
                            research_goal: ResearchGoal) -> Hypothesis:
        """
        Combine multiple hypotheses into a new, improved hypothesis.
        
        Args:
            hypotheses (List[Hypothesis]): The hypotheses to combine.
            research_goal (ResearchGoal): The research goal.
            
        Returns:
            Hypothesis: The combined hypothesis.
        """
        logger.info(f"Combining {len(hypotheses)} hypotheses")
        
        # Ensure we have at least two hypotheses to combine
        if len(hypotheses) < 2:
            logger.warning("Need at least two hypotheses to combine")
            return hypotheses[0] if hypotheses else None
        
        # Build the prompt
        hypotheses_text = "\n\n".join([
            f"Hypothesis {i+1}:\nTitle: {h.title}\nSummary: {h.summary}\nDescription: {h.description}\nSupporting Evidence: {', '.join(h.supporting_evidence)}"
            for i, h in enumerate(hypotheses)
        ])
        
        prompt = f"""
        You are combining multiple scientific hypotheses into a new, stronger hypothesis that addresses the research goal.
        
        Research Goal:
        {research_goal.text}
        
        Existing Hypotheses:
        {hypotheses_text}
        
        Your task is to create a new hypothesis that:
        1. Combines the strongest elements from each input hypothesis
        2. Resolves any contradictions between the input hypotheses
        3. Creates a synthesis that is more powerful than any individual hypothesis
        4. Is novel, coherent, and directly addresses the research goal
        5. Is scientifically sound and testable
        
        This is not simply a summary; it should be a novel synthesis that represents a conceptual advancement.
        
        Format your response as a JSON object with the following structure:
        
        ```json
        {{
            "title": "Concise title for the combined hypothesis",
            "summary": "Brief summary (1-2 sentences)",
            "description": "Detailed description of the combined hypothesis (1-2 paragraphs)",
            "supporting_evidence": ["Evidence 1", "Evidence 2", ...],
            "elements_used": [
                {{
                    "hypothesis": 1,
                    "elements": ["Element 1 from hypothesis 1", "Element 2 from hypothesis 1", ...]
                }},
                ...
            ],
            "synergy": "Explanation of how the combination creates value beyond the individual hypotheses"
        }}
        ```
        """
        
        # Generate combined hypothesis
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
            
            # Create the combined hypothesis
            combined = Hypothesis(
                title=data["title"],
                description=data["description"],
                summary=data["summary"],
                supporting_evidence=data["supporting_evidence"],
                creator="evolution_combine",
                parent_hypotheses=[h.id for h in hypotheses],
                metadata={
                    "research_goal_id": research_goal.id,
                    "elements_used": data.get("elements_used", []),
                    "synergy": data.get("synergy", "")
                }
            )
            
            logger.info(f"Created combined hypothesis: {combined.title}")
            return combined
            
        except Exception as e:
            logger.error(f"Error parsing combined hypothesis from response: {e}")
            logger.debug(f"Response: {response}")
            
            # Return the highest-rated hypothesis in case of error
            return max(hypotheses, key=lambda h: h.elo_rating)
    
    async def generate_out_of_box_hypothesis(self, 
                                        research_goal: ResearchGoal,
                                        existing_hypotheses: List[Hypothesis] = None) -> Hypothesis:
        """
        Generate an out-of-the-box hypothesis that takes a different approach.
        
        Args:
            research_goal (ResearchGoal): The research goal.
            existing_hypotheses (List[Hypothesis]): Existing hypotheses to diverge from.
            
        Returns:
            Hypothesis: The out-of-the-box hypothesis.
        """
        logger.info(f"Generating out-of-the-box hypothesis for research goal {research_goal.id}")
        
        # Prepare context of existing hypotheses if provided
        existing_context = ""
        if existing_hypotheses:
            top_hypotheses = sorted(existing_hypotheses, key=lambda h: h.elo_rating, reverse=True)[:3]
            
            hypotheses_text = "\n\n".join([
                f"Hypothesis {i+1}:\nTitle: {h.title}\nSummary: {h.summary}"
                for i, h in enumerate(top_hypotheses)
            ])
            
            existing_context = f"""
            Current Top Hypotheses (to diverge from):
            {hypotheses_text}
            """
        
        # Build the prompt
        prompt = f"""
        You are generating an out-of-the-box, creative scientific hypothesis for the following research goal:
        
        Research Goal:
        {research_goal.text}
        
        {existing_context}
        
        Your task is to create a novel hypothesis that:
        1. Takes a radically different approach from conventional thinking and existing hypotheses
        2. Challenges fundamental assumptions in the field
        3. Draws inspiration from other scientific disciplines or paradigm shifts
        4. Is still scientifically plausible and testable
        5. Directly addresses the research goal
        
        The hypothesis should be creative and unconventional but still scientifically sound - not pseudoscience.
        
        Format your response as a JSON object with the following structure:
        
        ```json
        {{
            "title": "Concise title for the out-of-the-box hypothesis",
            "summary": "Brief summary (1-2 sentences)",
            "description": "Detailed description of the hypothesis (1-2 paragraphs)",
            "supporting_evidence": ["Evidence 1", "Evidence 2", ...],
            "unconventional_aspects": ["Aspect 1", "Aspect 2", ...],
            "inspiration": "What inspired this unconventional approach"
        }}
        ```
        """
        
        # Generate out-of-the-box hypothesis
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
            
            # Create the out-of-the-box hypothesis
            hypothesis = Hypothesis(
                title=data["title"],
                description=data["description"],
                summary=data["summary"],
                supporting_evidence=data["supporting_evidence"],
                creator="evolution_out_of_box",
                tags={"out_of_box", "creative"},
                metadata={
                    "research_goal_id": research_goal.id,
                    "unconventional_aspects": data.get("unconventional_aspects", []),
                    "inspiration": data.get("inspiration", "")
                }
            )
            
            logger.info(f"Created out-of-the-box hypothesis: {hypothesis.title}")
            return hypothesis
            
        except Exception as e:
            logger.error(f"Error parsing out-of-the-box hypothesis from response: {e}")
            logger.debug(f"Response: {response}")
            
            # Create a basic hypothesis in case of error
            hypothesis = Hypothesis(
                title=f"Creative approach to {research_goal.text[:50]}...",
                description="Error parsing generated hypothesis.",
                summary="Error parsing generated hypothesis.",
                supporting_evidence=["Error during generation"],
                creator="evolution_out_of_box",
                tags={"out_of_box", "error"},
                metadata={
                    "research_goal_id": research_goal.id,
                    "generation_error": str(e)
                }
            )
            
            return hypothesis
    
    async def simplify_hypothesis(self, 
                             hypothesis: Hypothesis, 
                             research_goal: ResearchGoal) -> Hypothesis:
        """
        Simplify a complex hypothesis while preserving its core idea.
        
        Args:
            hypothesis (Hypothesis): The hypothesis to simplify.
            research_goal (ResearchGoal): The research goal.
            
        Returns:
            Hypothesis: The simplified hypothesis.
        """
        logger.info(f"Simplifying hypothesis {hypothesis.id}: {hypothesis.title}")
        
        # Build the prompt
        prompt = f"""
        You are simplifying a complex scientific hypothesis while preserving its core idea and scientific validity.
        
        Research Goal:
        {research_goal.text}
        
        Original Hypothesis:
        Title: {hypothesis.title}
        Summary: {hypothesis.summary}
        Description: {hypothesis.description}
        Supporting Evidence: {', '.join(hypothesis.supporting_evidence)}
        
        Your task is to create a simplified version of this hypothesis that:
        1. Preserves the essential scientific idea and mechanism
        2. Reduces unnecessary complexity and jargon
        3. Makes the hypothesis more accessible and testable
        4. Maintains scientific rigor and alignment with the research goal
        5. Is clearer and more concise
        
        This is not about dumbing down the science, but rather about expressing it more elegantly and clearly.
        
        Format your response as a JSON object with the following structure:
        
        ```json
        {{
            "title": "Simplified title for the hypothesis",
            "summary": "Brief simplified summary (1-2 sentences)",
            "description": "Simplified description of the hypothesis (1-2 paragraphs)",
            "supporting_evidence": ["Evidence 1", "Evidence 2", ...],
            "simplifications_made": ["Simplification 1", "Simplification 2", ...],
            "preserved_elements": ["Essential element 1 preserved", "Essential element 2 preserved", ...]
        }}
        ```
        """
        
        # Generate simplified hypothesis
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
            
            # Create the simplified hypothesis
            simplified = Hypothesis(
                title=data["title"],
                description=data["description"],
                summary=data["summary"],
                supporting_evidence=data["supporting_evidence"],
                creator="evolution_simplify",
                parent_hypotheses=[hypothesis.id],
                tags={"simplified"},
                metadata={
                    "research_goal_id": research_goal.id,
                    "simplifications_made": data.get("simplifications_made", []),
                    "preserved_elements": data.get("preserved_elements", [])
                }
            )
            
            logger.info(f"Created simplified hypothesis: {simplified.title}")
            return simplified
            
        except Exception as e:
            logger.error(f"Error parsing simplified hypothesis from response: {e}")
            logger.debug(f"Response: {response}")
            
            # Return the original hypothesis in case of error
            return hypothesis