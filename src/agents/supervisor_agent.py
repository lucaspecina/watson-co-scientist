"""
Supervisor Agent for coordinating the work of specialized agents.
"""

import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Set, Tuple
import random

from .base_agent import BaseAgent
from ..config.config import SystemConfig
from ..core.models import (
    Hypothesis, 
    ResearchGoal,
    Review,
    TournamentMatch,
    ResearchOverview,
    MetaReview,
    ExperimentalProtocol
)

logger = logging.getLogger("co_scientist")

class SupervisorAgent(BaseAgent):
    """
    Agent responsible for coordinating the work of specialized agents and managing system resources.
    """
    
    def __init__(self, config: SystemConfig):
        """
        Initialize the Supervisor agent.
        
        Args:
            config (SystemConfig): The system configuration.
        """
        super().__init__("supervisor", config)
    
    async def parse_research_goal(self, research_goal_text: str) -> Dict[str, Any]:
        """
        Parse a research goal text into a structured configuration.
        
        Args:
            research_goal_text (str): The research goal text.
            
        Returns:
            Dict[str, Any]: The research plan configuration.
        """
        logger.info(f"Parsing research goal: {research_goal_text[:100]}...")
        
        # Build the prompt
        prompt = f"""
        You are parsing a scientific research goal to extract key information and preferences that will guide the hypothesis generation and evaluation process.
        
        Research Goal:
        {research_goal_text}
        
        Your task is to:
        1. Identify the main objective and scope of the research goal
        2. Extract any constraints or requirements specified
        3. Determine preferences for hypothesis generation (novelty, practical application, etc.)
        4. Identify relevant scientific domains and subfields
        5. Determine evaluation criteria for hypotheses beyond the default criteria
        
        Format your response as a JSON object with the following structure:
        
        ```json
        {{
            "main_objective": "Concise statement of the main research objective",
            "scope": "The scope or boundaries of the research",
            "constraints": ["Constraint 1", "Constraint 2", ...],
            "preferences": {{
                "novelty_required": true/false,
                "practical_application_focus": true/false,
                "interdisciplinary_approach": true/false,
                "time_sensitivity": "short-term", "medium-term", or "long-term"
            }},
            "domains": ["Scientific domain 1", "Scientific domain 2", ...],
            "evaluation_criteria": ["Criterion 1", "Criterion 2", ...]
        }}
        ```
        
        Analyze the research goal carefully to extract all relevant information that would help guide the research process.
        """
        
        # Generate research plan configuration
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
            
            logger.info(f"Successfully parsed research goal into plan configuration")
            return data
            
        except Exception as e:
            logger.error(f"Error parsing research plan configuration from response: {e}")
            logger.debug(f"Response: {response}")
            
            # Return a basic configuration in case of error
            return {
                "main_objective": research_goal_text[:100],
                "scope": "Unable to determine scope",
                "constraints": [],
                "preferences": {
                    "novelty_required": True,
                    "practical_application_focus": True,
                    "interdisciplinary_approach": True,
                    "time_sensitivity": "medium-term"
                },
                "domains": ["Unable to determine domains"],
                "evaluation_criteria": []
            }
    
    async def plan_task_allocation(self, 
                              research_goal: ResearchGoal,
                              plan_config: Dict[str, Any],
                              current_state: Dict[str, Any]) -> Dict[str, float]:
        """
        Plan the allocation of tasks to specialized agents.
        
        Args:
            research_goal (ResearchGoal): The research goal.
            plan_config (Dict[str, Any]): The research plan configuration.
            current_state (Dict[str, Any]): The current state of the system.
            
        Returns:
            Dict[str, float]: The agent weights/priorities.
        """
        logger.info(f"Planning task allocation for research goal {research_goal.id}")
        
        # Get user feedback and research focus areas
        recent_feedback = []
        active_focus_areas = []
        user_hypotheses_count = 0
        
        try:
            # Access system data safely through class-level attributes
            from ..core.system import CoScientistSystem
            
            # Find the system instance
            system = None
            # If this module has a global 'system' variable
            import sys
            main_module = sys.modules.get('__main__')
            if main_module and hasattr(main_module, 'system') and isinstance(main_module.system, CoScientistSystem):
                system = main_module.system
            
            # Get database access
            db = system.db if system else None
            
            if db:
                # Get recent user feedback
                recent_feedback = db.get_user_feedback(research_goal.id, limit=5)
                
                # Get active research focus areas
                active_focus_areas = db.get_active_research_focus(research_goal.id)
                
                # Count user-submitted hypotheses
                all_hypotheses = db.hypotheses.get_all()
                user_hypotheses_count = sum(1 for h in all_hypotheses if getattr(h, 'source', '') == 'user' and 
                                           h.metadata.get("research_goal_id") == research_goal.id)
        except Exception as e:
            logger.warning(f"Error fetching user data for task allocation: {e}")
        
        # Format user data for the prompt
        feedback_text = ""
        if recent_feedback:
            feedback_text = "Recent User Feedback:\n"
            for i, feedback in enumerate(recent_feedback, 1):
                feedback_text += f"{i}. Type: {feedback.feedback_type}, User: {feedback.user_id}\n"
                feedback_text += f"   {feedback.text}\n\n"
        
        focus_text = ""
        if active_focus_areas:
            focus_text = "Active Research Focus Areas:\n"
            for i, focus in enumerate(active_focus_areas, 1):
                focus_text += f"{i}. Title: {focus.title}, Priority: {focus.priority}\n"
                focus_text += f"   {focus.description}\n"
                if focus.keywords:
                    focus_text += f"   Keywords: {', '.join(focus.keywords)}\n\n"
        
        user_hypotheses_text = ""
        if user_hypotheses_count > 0:
            user_hypotheses_text = f"There are {user_hypotheses_count} user-submitted hypotheses that need to be evaluated.\n\n"
        
        # Check if we need to generate experimental protocols
        protocols_count = 0
        hypotheses_without_protocols = 0
        
        try:
            if system and db:
                # Count experimental protocols
                protocols_count = len(db.experimental_protocols.get_all())
                
                # Count hypotheses that don't have protocols
                all_protocols = db.experimental_protocols.get_all()
                protocol_hypothesis_ids = set(p.hypothesis_id for p in all_protocols)
                
                all_hypotheses = db.get_hypotheses_for_goal(research_goal.id)
                reviewed_hypotheses = [h for h in all_hypotheses if h.status in ["reviewed", "accepted"]]
                
                hypotheses_without_protocols = sum(1 for h in reviewed_hypotheses if h.id not in protocol_hypothesis_ids)
        except Exception as e:
            logger.warning(f"Error checking protocol status: {e}")
            
        protocol_need_text = ""
        if hypotheses_without_protocols > 0:
            protocol_need_text = f"There are {hypotheses_without_protocols} reviewed hypotheses that don't have experimental protocols. Experimental protocols should be prioritized.\n\n"
        
        # Build the prompt
        prompt = f"""
        You are the Supervisor agent responsible for allocating computational resources and prioritizing tasks for specialized agents in the Co-Scientist system.
        
        Research Goal:
        {research_goal.text}
        
        Research Plan Configuration:
        {json.dumps(plan_config, indent=2)}
        
        Current System State:
        {json.dumps(current_state, indent=2)}
        
        {feedback_text}
        {focus_text}
        {user_hypotheses_text}
        {protocol_need_text}
        
        The Co-Scientist system has the following specialized agents:
        1. Generation agent: Generates novel research hypotheses and experimental protocols
        2. Reflection agent: Reviews and evaluates hypotheses and experimental protocols
        3. Ranking agent: Conducts tournaments to rank hypotheses
        4. Proximity agent: Calculates similarity between hypotheses
        5. Evolution agent: Improves existing hypotheses
        6. Meta-review agent: Synthesizes insights from reviews and tournaments
        
        Based on the current state, research goal, and user input, your task is to allocate computational resources (expressed as weights from 0.0 to 1.0) to each agent for the next iteration.
        
        Format your response as a JSON object with the following structure:
        
        ```json
        {{
            "generation": 0.0-1.0,
            "reflection": 0.0-1.0,
            "ranking": 0.0-1.0,
            "proximity": 0.0-1.0,
            "evolution": 0.0-1.0,
            "meta_review": 0.0-1.0,
            "protocol_generation": 0.0-1.0,
            "rationale": "Explanation for the weights assigned"
        }}
        ```
        
        Consider the following guidelines:
        - If there are few hypotheses, prioritize the Generation agent
        - If there are many unreviewed hypotheses, prioritize the Reflection agent
        - Once sufficient hypotheses are reviewed, prioritize the Ranking agent to establish relative quality
        - After several tournaments, prioritize the Evolution agent to improve top hypotheses
        - The Proximity agent is useful periodically to organize hypotheses
        - The Meta-review agent is more valuable later in the process
        - If there are user-submitted hypotheses, prioritize the Reflection and Ranking agents to evaluate them
        - If there are active research focus areas, prioritize Generation and Evolution agents to explore those areas
        - If there is recent user feedback, adjust weights to address the feedback appropriately
        - If there are reviewed hypotheses without experimental protocols, allocate resources to protocol generation
        - Protocol generation (performed by the Generation agent) should be prioritized after hypotheses have been reviewed and ranked
        
        The weights should sum to approximately 1.0 and reflect your judgment of how to allocate resources for the next iteration.
        """
        
        # Generate agent weights
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
            
            # Extract the weights
            weights = {
                "generation": float(data["generation"]),
                "reflection": float(data["reflection"]),
                "ranking": float(data["ranking"]),
                "proximity": float(data["proximity"]),
                "evolution": float(data["evolution"]),
                "meta_review": float(data["meta_review"]),
                "protocol_generation": float(data.get("protocol_generation", 0.0))
            }
            
            # Ensure no component gets zero weight - minimum 0.1 for all components
            for key in weights:
                weights[key] = max(weights[key], 0.1)
                
            # Add a small amount of randomness to ensure weights change between iterations
            for key in weights:
                # Add random variation of ±10% to each weight
                variation = random.uniform(-0.1, 0.1)
                weights[key] = max(0.1, weights[key] + variation)
            
            # Normalize the weights to ensure they sum to 1.0
            total = sum(weights.values())
            if total > 0:
                weights = {k: v / total for k, v in weights.items()}
            
            logger.info(f"Planned task allocation: {weights}")
            return weights
            
        except Exception as e:
            logger.error(f"Error parsing task allocation from response: {e}")
            logger.debug(f"Response: {response}")
            
            # Return default weights in case of error
            num_hypotheses = current_state.get("num_hypotheses", 0)
            num_reviews = current_state.get("num_reviews", 0)
            num_matches = current_state.get("num_tournament_matches", 0)
            
            # Simple logic for default weights
            if num_hypotheses < 10:
                # Early phase: focus on generation
                weights = {
                    "generation": 0.65,
                    "reflection": 0.2,
                    "ranking": 0.05,
                    "proximity": 0.1,
                    "evolution": 0.1,
                    "meta_review": 0.1,
                    "protocol_generation": 0.1
                }
            elif num_reviews < num_hypotheses * 0.5:
                # Middle phase: focus on review
                weights = {
                    "generation": 0.25,
                    "reflection": 0.45,
                    "ranking": 0.1,
                    "proximity": 0.1, 
                    "evolution": 0.1,
                    "meta_review": 0.1,
                    "protocol_generation": 0.1
                }
            elif num_matches < 20:
                # Tournament phase: focus on ranking
                weights = {
                    "generation": 0.1,
                    "reflection": 0.2,
                    "ranking": 0.3,
                    "proximity": 0.1,
                    "evolution": 0.1,
                    "meta_review": 0.1,
                    "protocol_generation": 0.1
                }
            else:
                # Late phase: focus on evolution, protocols, and meta-review
                weights = {
                    "generation": 0.1,
                    "reflection": 0.1,
                    "ranking": 0.15,
                    "proximity": 0.1,
                    "evolution": 0.25,
                    "meta_review": 0.15,
                    "protocol_generation": 0.2
                }
            
            # Add randomness to ensure weights change between iterations
            for key in weights:
                variation = random.uniform(-0.05, 0.05)
                weights[key] = max(0.1, weights[key] + variation)
                
            # Normalize to ensure they sum to 1.0
            total = sum(weights.values())
            if total > 0:
                weights = {k: v / total for k, v in weights.items()}
            
            return weights
    
    async def evaluate_termination(self, 
                              research_goal: ResearchGoal,
                              current_state: Dict[str, Any],
                              plan_config: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Evaluate whether the system should terminate its computation.
        
        Args:
            research_goal (ResearchGoal): The research goal.
            current_state (Dict[str, Any]): The current state of the system.
            plan_config (Dict[str, Any]): The research plan configuration.
            
        Returns:
            Tuple[bool, str]: Whether to terminate and the reason.
        """
        logger.info(f"Evaluating termination for research goal {research_goal.id}")
        
        # Build the prompt
        prompt = f"""
        You are the Supervisor agent responsible for deciding when the Co-Scientist system should terminate its computation for a given research goal.
        
        Research Goal:
        {research_goal.text}
        
        Research Plan Configuration:
        {json.dumps(plan_config, indent=2)}
        
        Current System State:
        {json.dumps(current_state, indent=2)}
        
        Your task is to determine whether the system should terminate its computation based on the following criteria:
        1. Whether the system has generated sufficient high-quality hypotheses
        2. Whether additional computation is likely to yield significant improvements
        3. Whether the system has satisfied the requirements specified in the research goal
        4. Whether the computational budget has been exhausted
        
        Format your response as a JSON object with the following structure:
        
        ```json
        {{
            "terminate": true or false,
            "reason": "Detailed explanation for the decision",
            "confidence": 0.0-1.0
        }}
        ```
        
        Provide a balanced assessment of whether the system should continue or terminate computation.
        """
        
        # Generate termination evaluation
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
            
            terminate = bool(data["terminate"])
            reason = data["reason"]
            
            logger.info(f"Termination evaluation: {terminate} (Reason: {reason})")
            return terminate, reason
            
        except Exception as e:
            logger.error(f"Error parsing termination evaluation from response: {e}")
            logger.debug(f"Response: {response}")
            
            # Default to not terminating in case of error
            return False, "Error evaluating termination condition"
            
    async def extract_keywords(self, text: str) -> List[str]:
        """
        Extract relevant keywords from a text for use in research focus areas.
        
        Args:
            text (str): The text to extract keywords from.
            
        Returns:
            List[str]: A list of relevant keywords.
        """
        prompt = f"""
        Extract the most important and relevant scientific keywords from the following text.
        These keywords will be used to index and categorize this research focus area.
        Focus on domain-specific technical terms, scientific concepts, and precise terminology.
        
        Text:
        {text}
        
        Return a JSON array of 5-10 keywords, ordered by relevance. Do not include any explanations,
        just the JSON array of keywords.
        
        Example response format:
        ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"]
        """
        
        response = await self.generate(prompt)
        
        try:
            # Extract JSON array
            if "[" in response and "]" in response:
                json_content = response[response.find("["):response.rfind("]")+1]
                keywords = json.loads(json_content)
                return keywords
            else:
                # Fall back to splitting by commas and cleanup if JSON parsing fails
                keywords = [k.strip().strip('"\'') for k in response.split(",")]
                return [k for k in keywords if k]
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            # Return some basic keywords extracted from the text
            words = text.split()
            return [w for w in words if len(w) > 4][:5]

    async def update_research_plan(self, 
                              research_goal: ResearchGoal,
                              plan_config: Dict[str, Any],
                              feedback_text: str) -> Dict[str, Any]:
        """
        Update the research plan based on user feedback.
        
        Args:
            research_goal (ResearchGoal): The research goal.
            plan_config (Dict[str, Any]): The current research plan configuration.
            feedback_text (str): The feedback text from the user.
            
        Returns:
            Dict[str, Any]: The updated research plan configuration.
        """
        logger.info(f"Updating research plan for goal {research_goal.id} based on feedback")
        
        # Build the prompt
        prompt = f"""
        You are the Supervisor agent responsible for updating the research plan based on user feedback.
        
        Research Goal:
        {research_goal.text}
        
        Current Research Plan Configuration:
        {json.dumps(plan_config, indent=2)}
        
        User Feedback:
        {feedback_text}
        
        Your task is to analyze the user feedback and recommend changes to the research plan configuration.
        Consider whether the feedback suggests specific topics to explore, methodologies to use, or constraints to apply.
        
        Format your response as a JSON object representing the UPDATED parts of the plan ONLY.
        Include only keys that should be changed or added based on the feedback.
        
        Example response format:
        ```json
        {{
            "preferences": {{
                "focus_areas": ["specific area mentioned in feedback"],
                "methodologies": ["methods suggested in feedback"]
            }},
            "constraints": ["any constraints mentioned in feedback"]
        }}
        ```
        
        If no changes are needed, return an empty JSON object: {{}}
        """
        
        # Generate the updated plan
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
            updates = json.loads(json_content)
            
            # If updates is empty, return an empty dict
            if not updates:
                logger.info("No updates needed for research plan")
                return {}
            
            # Log the updates
            logger.info(f"Updating research plan with: {updates}")
            return updates
            
        except Exception as e:
            logger.error(f"Error parsing research plan updates: {e}")
            logger.debug(f"Response: {response}")
            return {}