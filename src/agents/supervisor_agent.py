"""
Supervisor Agent for coordinating the work of specialized agents.
"""

import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Set, Tuple

from .base_agent import BaseAgent
from ..config.config import SystemConfig
from ..core.models import (
    Hypothesis, 
    ResearchGoal,
    Review,
    TournamentMatch,
    ResearchOverview,
    MetaReview
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
        
        The Co-Scientist system has the following specialized agents:
        1. Generation agent: Generates novel research hypotheses
        2. Reflection agent: Reviews and evaluates hypotheses
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
                "meta_review": float(data["meta_review"])
            }
            
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
                    "generation": 0.7,
                    "reflection": 0.2,
                    "ranking": 0.05,
                    "proximity": 0.0,
                    "evolution": 0.0,
                    "meta_review": 0.05
                }
            elif num_reviews < num_hypotheses * 0.5:
                # Middle phase: focus on review
                weights = {
                    "generation": 0.3,
                    "reflection": 0.5,
                    "ranking": 0.1,
                    "proximity": 0.05,
                    "evolution": 0.0,
                    "meta_review": 0.05
                }
            elif num_matches < 20:
                # Tournament phase: focus on ranking
                weights = {
                    "generation": 0.1,
                    "reflection": 0.2,
                    "ranking": 0.4,
                    "proximity": 0.1,
                    "evolution": 0.1,
                    "meta_review": 0.1
                }
            else:
                # Late phase: focus on evolution and meta-review
                weights = {
                    "generation": 0.1,
                    "reflection": 0.1,
                    "ranking": 0.2,
                    "proximity": 0.1,
                    "evolution": 0.3,
                    "meta_review": 0.2
                }
            
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