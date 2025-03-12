"""
Reflection agent for mini-RAUL.
This agent analyzes and reflects on research hypotheses.
"""
import logging
import json
from typing import Dict, Any, List, Optional

from ..core.agent import Agent
from ..models.llm import LLMProvider

logger = logging.getLogger(__name__)


class ReflectionAgent(Agent):
    """
    Reflection agent that analyzes research hypotheses.
    Provides detailed reviews, critiques, and verifications.
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the reflection agent.
        
        Args:
            name: The name of the agent.
            config: Optional configuration for the agent.
        """
        super().__init__(name, config)
        
        # Initialize LLM client
        llm_provider = self.config.get("llm_provider", "azure")
        llm_config = self.config.get("llm_config", {})
        self.llm = LLMProvider.get_client(llm_provider, llm_config)
        
        self.reflection_type = self.config.get("reflection_type", "full_review")
        self.temperature = self.config.get("temperature", 0.2)  # Low temperature for analytical tasks
        
    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze and reflect on research hypotheses.
        
        Args:
            context: The context information including hypotheses and rankings.
            
        Returns:
            Dict containing the reflection results.
        """
        research_goal = context.get("research_goal", "")
        hypothesis = context.get("hypothesis", {})
        hypotheses = context.get("hypotheses", [])
        rankings = context.get("rankings", [])
        
        if not research_goal:
            raise ValueError("Research goal is required")
        
        # Select reflection type based on configuration
        if self.reflection_type == "full_review":
            if not hypothesis:
                raise ValueError("Hypothesis is required for full review")
            result = await self._full_review(research_goal, hypothesis)
        elif self.reflection_type == "tournament_review":
            if not hypotheses or not rankings:
                raise ValueError("Hypotheses and rankings are required for tournament review")
            result = await self._tournament_review(research_goal, hypotheses, rankings)
        elif self.reflection_type == "deep_verification":
            if not hypothesis:
                raise ValueError("Hypothesis is required for deep verification")
            result = await self._deep_verification(research_goal, hypothesis)
        else:
            raise ValueError(f"Unsupported reflection type: {self.reflection_type}")
        
        return {
            "reflection": result,
            "reflection_type": self.reflection_type,
            "agent_name": self.name
        }
    
    async def _full_review(self, 
                         research_goal: str, 
                         hypothesis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform a full review of a hypothesis.
        
        Args:
            research_goal: The research goal.
            hypothesis: The hypothesis to review.
            
        Returns:
            The review results.
        """
        system_message = """
        You are an expert scientific reviewer conducting a thorough analysis of a research hypothesis.
        Your task is to analyze the hypothesis from multiple angles:
        1. Scientific validity and plausibility
        2. Novelty and originality
        3. Potential impact if proven correct
        4. Experimental feasibility and testability
        5. Strengths and limitations
        6. Alignment with the research goal
        
        Provide a balanced critique that highlights both strengths and weaknesses.
        """
        
        # Get the hypothesis content
        hypothesis_content = hypothesis.get("content", {}).get("hypothesis", {})
        
        # Convert to string if needed
        if isinstance(hypothesis_content, dict):
            hypothesis_str = json.dumps(hypothesis_content, indent=2)
        else:
            hypothesis_str = str(hypothesis_content)
        
        prompt = f"""
        # Research Goal
        {research_goal}
        
        # Hypothesis
        {hypothesis_str}
        
        Please provide a comprehensive review of this hypothesis.
        
        Format your response as a JSON object with the following structure:
        {{
            "summary": "Brief summary of the hypothesis",
            "scientific_validity": "Analysis of the scientific validity and plausibility",
            "novelty": "Assessment of novelty and originality",
            "potential_impact": "Evaluation of potential impact if proven correct",
            "experimental_feasibility": "Analysis of how feasible it would be to test experimentally",
            "strengths": ["List of key strengths"],
            "limitations": ["List of key limitations"],
            "alignment": "Assessment of how well the hypothesis addresses the research goal",
            "overall_assessment": "Overall assessment of the hypothesis"
        }}
        """
        
        try:
            response = await self.llm.generate(
                prompt=prompt,
                system_message=system_message,
                temperature=self.temperature
            )
            
            # Process the response to extract the JSON
            import re
            
            # Look for JSON content in the response
            json_match = re.search(r'(\{.*\})', response, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1)
                # Try to parse the JSON
                try:
                    result = json.loads(json_str)
                    return result
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse review JSON: {json_str}")
                    return {"error": "Failed to parse review", "raw_response": response}
            else:
                logger.error(f"No JSON content found in response: {response}")
                return {"error": "No JSON content found", "raw_response": response}
            
        except Exception as e:
            logger.error(f"Error generating review: {e}")
            return {"error": str(e)}
    
    async def _tournament_review(self, 
                               research_goal: str, 
                               hypotheses: List[Dict[str, Any]],
                               rankings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Review the results of a tournament ranking.
        
        Args:
            research_goal: The research goal.
            hypotheses: The hypotheses in the tournament.
            rankings: The ranking results.
            
        Returns:
            The tournament review.
        """
        system_message = """
        You are an expert scientific reviewer analyzing the results of a tournament-style ranking of research hypotheses.
        Your task is to:
        1. Identify patterns in the top-ranked hypotheses
        2. Analyze common strengths of successful hypotheses
        3. Identify common limitations in lower-ranked hypotheses
        4. Provide insights on what makes a hypothesis successful
        5. Suggest directions for improvement
        """
        
        # Create a mapping of hypothesis IDs to hypotheses
        hypotheses_map = {h.get("id", ""): h for h in hypotheses}
        
        # Sort rankings by rank
        sorted_rankings = sorted(rankings, key=lambda x: x.get("rank", 999))
        
        # Prepare top and bottom hypotheses for analysis
        top_hypotheses = []
        bottom_hypotheses = []
        
        for i, ranking in enumerate(sorted_rankings):
            h_id = ranking.get("id", "")
            if h_id in hypotheses_map:
                h = hypotheses_map[h_id]
                h_with_rank = {
                    "id": h_id,
                    "rank": ranking.get("rank", i + 1),
                    "wins": ranking.get("wins", 0),
                    "losses": ranking.get("losses", 0),
                    "content": h.get("content", {})
                }
                
                if i < len(sorted_rankings) // 2:
                    top_hypotheses.append(h_with_rank)
                else:
                    bottom_hypotheses.append(h_with_rank)
        
        prompt = f"""
        # Research Goal
        {research_goal}
        
        # Top-Ranked Hypotheses
        {json.dumps(top_hypotheses, indent=2)}
        
        # Lower-Ranked Hypotheses
        {json.dumps(bottom_hypotheses, indent=2)}
        
        Please analyze these tournament results and provide insights.
        
        Format your response as a JSON object with the following structure:
        {{
            "top_hypothesis_patterns": "Analysis of common patterns in top-ranked hypotheses",
            "common_strengths": ["List of common strengths in successful hypotheses"],
            "common_limitations": ["List of common limitations in lower-ranked hypotheses"],
            "success_factors": "Analysis of what makes a hypothesis successful in this context",
            "improvement_suggestions": ["Suggestions for improving future hypotheses"],
            "overall_analysis": "Overall analysis of the tournament results"
        }}
        """
        
        try:
            response = await self.llm.generate(
                prompt=prompt,
                system_message=system_message,
                temperature=self.temperature
            )
            
            # Process the response to extract the JSON
            import re
            
            # Look for JSON content in the response
            json_match = re.search(r'(\{.*\})', response, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1)
                # Try to parse the JSON
                try:
                    result = json.loads(json_str)
                    return result
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse tournament review JSON: {json_str}")
                    return {"error": "Failed to parse tournament review", "raw_response": response}
            else:
                logger.error(f"No JSON content found in response: {response}")
                return {"error": "No JSON content found", "raw_response": response}
            
        except Exception as e:
            logger.error(f"Error generating tournament review: {e}")
            return {"error": str(e)}
    
    async def _deep_verification(self, 
                               research_goal: str, 
                               hypothesis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform deep verification of a hypothesis.
        
        Args:
            research_goal: The research goal.
            hypothesis: The hypothesis to verify.
            
        Returns:
            The verification results.
        """
        system_message = """
        You are an expert scientific reviewer conducting a rigorous, deep verification of a research hypothesis.
        Your task is to:
        1. Identify all key assumptions in the hypothesis
        2. Critically evaluate each assumption based on scientific knowledge
        3. Assess the logical consistency of the hypothesis
        4. Identify potential contradictions with established scientific knowledge
        5. Evaluate the mechanistic details and their plausibility
        
        Be thorough, critical, and objective in your analysis.
        """
        
        # Get the hypothesis content
        hypothesis_content = hypothesis.get("content", {}).get("hypothesis", {})
        
        # Convert to string if needed
        if isinstance(hypothesis_content, dict):
            hypothesis_str = json.dumps(hypothesis_content, indent=2)
        else:
            hypothesis_str = str(hypothesis_content)
        
        prompt = f"""
        # Research Goal
        {research_goal}
        
        # Hypothesis
        {hypothesis_str}
        
        Please conduct a deep verification of this hypothesis, identifying and evaluating all key assumptions.
        
        Format your response as a JSON object with the following structure:
        {{
            "key_assumptions": [
                {{
                    "assumption": "Statement of assumption",
                    "evaluation": "Critical evaluation of this assumption based on scientific knowledge",
                    "confidence": "High/Medium/Low confidence in this assumption"
                }},
                // more assumptions...
            ],
            "logical_consistency": "Assessment of the logical consistency of the hypothesis",
            "mechanistic_plausibility": "Evaluation of the mechanistic details and their plausibility",
            "potential_contradictions": ["List of potential contradictions with established knowledge"],
            "verification_outcome": "Overall verification outcome (Well-supported/Partially supported/Weakly supported)"
        }}
        """
        
        try:
            response = await self.llm.generate(
                prompt=prompt,
                system_message=system_message,
                temperature=self.temperature
            )
            
            # Process the response to extract the JSON
            import re
            
            # Look for JSON content in the response
            json_match = re.search(r'(\{.*\})', response, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1)
                # Try to parse the JSON
                try:
                    result = json.loads(json_str)
                    return result
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse verification JSON: {json_str}")
                    return {"error": "Failed to parse verification", "raw_response": response}
            else:
                logger.error(f"No JSON content found in response: {response}")
                return {"error": "No JSON content found", "raw_response": response}
            
        except Exception as e:
            logger.error(f"Error generating verification: {e}")
            return {"error": str(e)} 