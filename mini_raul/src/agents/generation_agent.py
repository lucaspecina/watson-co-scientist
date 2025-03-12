"""
Generation agent for mini-RAUL.
This agent generates research hypotheses based on the research goal.
"""
import logging
from typing import Dict, Any, List, Optional

from ..core.agent import Agent
from ..models.llm import LLMProvider

logger = logging.getLogger(__name__)


class GenerationAgent(Agent):
    """
    Generation agent that produces research hypotheses.
    Can generate hypotheses through literature exploration or simulated debate.
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the generation agent.
        
        Args:
            name: The name of the agent.
            config: Optional configuration for the agent.
        """
        super().__init__(name, config)
        
        # Initialize LLM client
        llm_provider = self.config.get("llm_provider", "azure")
        llm_config = self.config.get("llm_config", {})
        self.llm = LLMProvider.get_client(llm_provider, llm_config)
        
        self.generation_type = self.config.get("generation_type", "literature_exploration")
        self.temperature = self.config.get("temperature", 0.7)
        
    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a research hypothesis.
        
        Args:
            context: The context information including research goal and constraints.
            
        Returns:
            Dict containing the generated hypothesis.
        """
        research_goal = context.get("research_goal", "")
        preferences = context.get("preferences", "")
        constraints = context.get("constraints", "")
        
        if not research_goal:
            raise ValueError("Research goal is required")
        
        # Build prompt based on generation type
        if self.generation_type == "literature_exploration":
            result = await self._generate_from_literature(research_goal, preferences, constraints)
        elif self.generation_type == "scientific_debate":
            result = await self._generate_from_debate(research_goal, preferences, constraints)
        else:
            raise ValueError(f"Unsupported generation type: {self.generation_type}")
        
        return {
            "hypothesis": result,
            "generation_type": self.generation_type,
            "agent_name": self.name
        }
    
    async def _generate_from_literature(self, 
                                      research_goal: str, 
                                      preferences: str,
                                      constraints: str) -> Dict[str, Any]:
        """
        Generate a hypothesis based on literature exploration.
        
        Args:
            research_goal: The research goal.
            preferences: User preferences.
            constraints: User constraints.
            
        Returns:
            The generated hypothesis.
        """
        system_message = """
        You are a research scientist generating a novel hypothesis for a given research goal.
        Your task is to generate a well-reasoned, scientifically plausible hypothesis based on your knowledge of the scientific literature.
        The hypothesis should be novel, testable, and address the research goal.
        Your response should include:
        1. A clear hypothesis statement
        2. Rationale based on existing literature
        3. Potential experimental approaches to test the hypothesis
        4. Predicted outcomes
        """
        
        prompt = f"""
        # Research Goal
        {research_goal}
        
        # Preferences
        {preferences}
        
        # Constraints
        {constraints}
        
        Please generate a novel, well-reasoned scientific hypothesis that addresses this research goal.
        
        Format your response as a JSON object with the following structure:
        {{
            "hypothesis_statement": "A clear, concise statement of your hypothesis",
            "rationale": "Detailed scientific reasoning supporting your hypothesis, citing relevant literature and concepts",
            "experimental_approach": "A description of how this hypothesis could be tested experimentally",
            "predicted_outcomes": "The expected results if your hypothesis is correct",
            "limitations": "Potential limitations or challenges to this hypothesis"
        }}
        """
        
        try:
            response = await self.llm.generate(
                prompt=prompt,
                system_message=system_message,
                temperature=self.temperature
            )
            
            # Process the response to extract the JSON
            # The response might have markdown or other formatting
            # We need to extract just the JSON part
            import re
            import json
            
            # Look for JSON content in the response
            json_match = re.search(r'(\{.*\})', response, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1)
                # Try to parse the JSON
                try:
                    result = json.loads(json_str)
                    return result
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse hypothesis JSON: {json_str}")
                    return {"error": "Failed to parse hypothesis", "raw_response": response}
            else:
                logger.error(f"No JSON content found in response: {response}")
                return {"error": "No JSON content found", "raw_response": response}
            
        except Exception as e:
            logger.error(f"Error generating hypothesis: {e}")
            return {"error": str(e)}
    
    async def _generate_from_debate(self, 
                                  research_goal: str, 
                                  preferences: str,
                                  constraints: str) -> Dict[str, Any]:
        """
        Generate a hypothesis based on simulated scientific debate.
        
        Args:
            research_goal: The research goal.
            preferences: User preferences.
            constraints: User constraints.
            
        Returns:
            The generated hypothesis.
        """
        system_message = """
        You are simulating a scientific debate between multiple experts to generate a novel hypothesis.
        Each expert will contribute their perspective on the research goal.
        From this debate, you will synthesize a novel, well-reasoned hypothesis.
        
        The experts are:
        1. Expert A: Molecular biologist with expertise in cellular pathways
        2. Expert B: Computational scientist with expertise in data analysis
        3. Expert C: Clinical researcher with expertise in translational applications
        
        Simulate their discussion and then synthesize a final hypothesis that integrates their perspectives.
        """
        
        prompt = f"""
        # Research Goal
        {research_goal}
        
        # Preferences
        {preferences}
        
        # Constraints
        {constraints}
        
        Please simulate a scientific debate between the experts and synthesize a novel hypothesis.
        
        Format your final response as a JSON object with the following structure:
        {{
            "debate_summary": "Summary of the key points discussed by the experts",
            "hypothesis_statement": "A clear, concise statement of the synthesized hypothesis",
            "rationale": "Detailed scientific reasoning supporting the hypothesis",
            "experimental_approach": "A description of how this hypothesis could be tested experimentally",
            "predicted_outcomes": "The expected results if the hypothesis is correct",
            "limitations": "Potential limitations or challenges to this hypothesis"
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
            import json
            
            # Look for JSON content in the response
            json_match = re.search(r'(\{.*\})', response, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1)
                # Try to parse the JSON
                try:
                    result = json.loads(json_str)
                    return result
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse hypothesis JSON: {json_str}")
                    return {"error": "Failed to parse hypothesis", "raw_response": response}
            else:
                logger.error(f"No JSON content found in response: {response}")
                return {"error": "No JSON content found", "raw_response": response}
            
        except Exception as e:
            logger.error(f"Error generating hypothesis: {e}")
            return {"error": str(e)} 