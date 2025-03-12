"""
Evolution agent for mini-RAUL.
This agent evolves and improves research hypotheses.
"""
import logging
import json
from typing import Dict, Any, List, Optional

from ..core.agent import Agent
from ..models.llm import LLMProvider

logger = logging.getLogger(__name__)


class EvolutionAgent(Agent):
    """
    Evolution agent that improves and evolves research hypotheses.
    Can create improved versions of hypotheses based on feedback and reviews.
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the evolution agent.
        
        Args:
            name: The name of the agent.
            config: Optional configuration for the agent.
        """
        super().__init__(name, config)
        
        # Initialize LLM client
        llm_provider = self.config.get("llm_provider", "azure")
        llm_config = self.config.get("llm_config", {})
        self.llm = LLMProvider.get_client(llm_provider, llm_config)
        
        self.evolution_type = self.config.get("evolution_type", "improvement")
        self.temperature = self.config.get("temperature", 0.7)
        
    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evolve and improve research hypotheses.
        
        Args:
            context: The context information including hypotheses and feedback.
            
        Returns:
            Dict containing the evolved hypothesis.
        """
        research_goal = context.get("research_goal", "")
        hypothesis = context.get("hypothesis", {})
        reflection = context.get("reflection", {})
        user_feedback = context.get("user_feedback", "")
        
        if not research_goal:
            raise ValueError("Research goal is required")
        
        if not hypothesis:
            raise ValueError("Hypothesis is required")
        
        # Select evolution type based on configuration
        if self.evolution_type == "improvement":
            result = await self._improve_hypothesis(research_goal, hypothesis, reflection, user_feedback)
        elif self.evolution_type == "simplification":
            result = await self._simplify_hypothesis(research_goal, hypothesis, reflection, user_feedback)
        elif self.evolution_type == "extension":
            result = await self._extend_hypothesis(research_goal, hypothesis, reflection, user_feedback)
        else:
            raise ValueError(f"Unsupported evolution type: {self.evolution_type}")
        
        return {
            "evolved_hypothesis": result,
            "evolution_type": self.evolution_type,
            "agent_name": self.name
        }
    
    async def _improve_hypothesis(self, 
                                research_goal: str, 
                                hypothesis: Dict[str, Any],
                                reflection: Dict[str, Any],
                                user_feedback: str) -> Dict[str, Any]:
        """
        Improve a hypothesis based on feedback and reflection.
        
        Args:
            research_goal: The research goal.
            hypothesis: The hypothesis to improve.
            reflection: Reflection on the hypothesis.
            user_feedback: User feedback.
            
        Returns:
            The improved hypothesis.
        """
        system_message = """
        You are an expert scientific researcher tasked with improving a research hypothesis.
        Your goal is to create an improved version of the hypothesis that addresses limitations and incorporates feedback.
        The improvements should be based on:
        1. Expert reflection on the original hypothesis
        2. User feedback (if provided)
        3. Your scientific judgment
        
        Create a significantly improved version while maintaining the core insight of the original hypothesis.
        """
        
        # Get the hypothesis content
        hypothesis_content = hypothesis.get("content", {}).get("hypothesis", {})
        
        # Convert to string if needed
        if isinstance(hypothesis_content, dict):
            hypothesis_str = json.dumps(hypothesis_content, indent=2)
        else:
            hypothesis_str = str(hypothesis_content)
        
        # Convert reflection to string if needed
        reflection_str = ""
        if reflection:
            if isinstance(reflection, dict):
                reflection_str = json.dumps(reflection, indent=2)
            else:
                reflection_str = str(reflection)
        
        prompt = f"""
        # Research Goal
        {research_goal}
        
        # Original Hypothesis
        {hypothesis_str}
        
        # Expert Reflection
        {reflection_str}
        
        # User Feedback
        {user_feedback}
        
        Please create an improved version of this hypothesis that addresses limitations and incorporates feedback.
        
        Format your response as a JSON object with the following structure:
        {{
            "hypothesis_statement": "A clear, concise statement of your improved hypothesis",
            "key_improvements": ["List of specific improvements made"],
            "rationale": "Detailed scientific reasoning supporting your improved hypothesis",
            "experimental_approach": "A description of how this improved hypothesis could be tested experimentally",
            "predicted_outcomes": "The expected results if your improved hypothesis is correct",
            "limitations": "Potential limitations or challenges of this improved hypothesis"
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
                    logger.error(f"Failed to parse improved hypothesis JSON: {json_str}")
                    return {"error": "Failed to parse improved hypothesis", "raw_response": response}
            else:
                logger.error(f"No JSON content found in response: {response}")
                return {"error": "No JSON content found", "raw_response": response}
            
        except Exception as e:
            logger.error(f"Error generating improved hypothesis: {e}")
            return {"error": str(e)}
    
    async def _simplify_hypothesis(self, 
                                 research_goal: str, 
                                 hypothesis: Dict[str, Any],
                                 reflection: Dict[str, Any],
                                 user_feedback: str) -> Dict[str, Any]:
        """
        Simplify a hypothesis while maintaining its core insight.
        
        Args:
            research_goal: The research goal.
            hypothesis: The hypothesis to simplify.
            reflection: Reflection on the hypothesis.
            user_feedback: User feedback.
            
        Returns:
            The simplified hypothesis.
        """
        system_message = """
        You are an expert scientific researcher tasked with simplifying a research hypothesis.
        Your goal is to create a simplified version that retains the core insight but:
        1. Reduces unnecessary complexity
        2. Makes the hypothesis more testable
        3. Focuses on the most essential mechanisms
        4. Uses clearer and more concise language
        
        Create a simplified version while preserving the original scientific value.
        """
        
        # Get the hypothesis content
        hypothesis_content = hypothesis.get("content", {}).get("hypothesis", {})
        
        # Convert to string if needed
        if isinstance(hypothesis_content, dict):
            hypothesis_str = json.dumps(hypothesis_content, indent=2)
        else:
            hypothesis_str = str(hypothesis_content)
        
        # Convert reflection to string if needed
        reflection_str = ""
        if reflection:
            if isinstance(reflection, dict):
                reflection_str = json.dumps(reflection, indent=2)
            else:
                reflection_str = str(reflection)
        
        prompt = f"""
        # Research Goal
        {research_goal}
        
        # Original Hypothesis
        {hypothesis_str}
        
        # Expert Reflection
        {reflection_str}
        
        # User Feedback
        {user_feedback}
        
        Please create a simplified version of this hypothesis that retains the core insight but reduces complexity.
        
        Format your response as a JSON object with the following structure:
        {{
            "simplified_statement": "A clear, concise statement of your simplified hypothesis",
            "simplification_changes": ["List of specific simplifications made"],
            "core_insight_preserved": "Explanation of how the core insight has been preserved",
            "experimental_approach": "A simplified experimental approach to test this hypothesis",
            "advantages_of_simplification": ["List of advantages gained through simplification"]
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
                    logger.error(f"Failed to parse simplified hypothesis JSON: {json_str}")
                    return {"error": "Failed to parse simplified hypothesis", "raw_response": response}
            else:
                logger.error(f"No JSON content found in response: {response}")
                return {"error": "No JSON content found", "raw_response": response}
            
        except Exception as e:
            logger.error(f"Error generating simplified hypothesis: {e}")
            return {"error": str(e)}
    
    async def _extend_hypothesis(self, 
                               research_goal: str, 
                               hypothesis: Dict[str, Any],
                               reflection: Dict[str, Any],
                               user_feedback: str) -> Dict[str, Any]:
        """
        Extend a hypothesis to explore new dimensions or applications.
        
        Args:
            research_goal: The research goal.
            hypothesis: The hypothesis to extend.
            reflection: Reflection on the hypothesis.
            user_feedback: User feedback.
            
        Returns:
            The extended hypothesis.
        """
        system_message = """
        You are an expert scientific researcher tasked with extending a research hypothesis.
        Your goal is to create an extended version that:
        1. Explores new dimensions or applications of the original hypothesis
        2. Broadens the scope or generalizability
        3. Connects to additional scientific domains
        4. Identifies novel implications
        
        Create an extended version that builds on the original while adding significant new value.
        """
        
        # Get the hypothesis content
        hypothesis_content = hypothesis.get("content", {}).get("hypothesis", {})
        
        # Convert to string if needed
        if isinstance(hypothesis_content, dict):
            hypothesis_str = json.dumps(hypothesis_content, indent=2)
        else:
            hypothesis_str = str(hypothesis_content)
        
        # Convert reflection to string if needed
        reflection_str = ""
        if reflection:
            if isinstance(reflection, dict):
                reflection_str = json.dumps(reflection, indent=2)
            else:
                reflection_str = str(reflection)
        
        prompt = f"""
        # Research Goal
        {research_goal}
        
        # Original Hypothesis
        {hypothesis_str}
        
        # Expert Reflection
        {reflection_str}
        
        # User Feedback
        {user_feedback}
        
        Please create an extended version of this hypothesis that explores new dimensions or applications.
        
        Format your response as a JSON object with the following structure:
        {{
            "extended_statement": "A clear statement of your extended hypothesis",
            "new_dimensions": ["List of new dimensions or applications explored"],
            "connections_to_other_domains": ["Scientific domains now connected to this hypothesis"],
            "novel_implications": "Novel implications of this extended hypothesis",
            "additional_experimental_approaches": ["New experimental approaches enabled by this extension"]
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
                    logger.error(f"Failed to parse extended hypothesis JSON: {json_str}")
                    return {"error": "Failed to parse extended hypothesis", "raw_response": response}
            else:
                logger.error(f"No JSON content found in response: {response}")
                return {"error": "No JSON content found", "raw_response": response}
            
        except Exception as e:
            logger.error(f"Error generating extended hypothesis: {e}")
            return {"error": str(e)} 