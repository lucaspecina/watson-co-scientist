"""
Meta-review agent for mini-RAUL.
This agent synthesizes research results and produces overviews.
"""
import logging
import json
from typing import Dict, Any, List, Optional

from ..core.agent import Agent
from ..models.llm import LLMProvider

logger = logging.getLogger(__name__)


class MetaReviewAgent(Agent):
    """
    Meta-review agent that synthesizes research results.
    Produces research overviews and summaries.
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the meta-review agent.
        
        Args:
            name: The name of the agent.
            config: Optional configuration for the agent.
        """
        super().__init__(name, config)
        
        # Initialize LLM client
        llm_provider = self.config.get("llm_provider", "azure")
        llm_config = self.config.get("llm_config", {})
        self.llm = LLMProvider.get_client(llm_provider, llm_config)
        
        self.temperature = self.config.get("temperature", 0.3)  # Lower temperature for more factual output
        
    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synthesize research results and create an overview.
        
        Args:
            context: The context information including research results.
            
        Returns:
            Dict containing the meta-review results.
        """
        research_goal = context.get("research_goal", "")
        hypotheses = context.get("hypotheses", [])
        rankings = context.get("rankings", [])
        reflections = context.get("reflections", [])
        evolutions = context.get("evolutions", [])
        iterations = context.get("iterations", [])
        
        if not research_goal:
            raise ValueError("Research goal is required")
        
        if not hypotheses:
            raise ValueError("At least one hypothesis is required")
        
        result = await self._create_research_overview(
            research_goal=research_goal,
            hypotheses=hypotheses,
            rankings=rankings,
            reflections=reflections,
            evolutions=evolutions,
            iterations=iterations
        )
        
        return {
            "meta_review": result,
            "agent_name": self.name
        }
    
    async def _create_research_overview(self, 
                                      research_goal: str, 
                                      hypotheses: List[Dict[str, Any]],
                                      rankings: List[Dict[str, Any]],
                                      reflections: List[Dict[str, Any]],
                                      evolutions: List[Dict[str, Any]],
                                      iterations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create a comprehensive research overview.
        
        Args:
            research_goal: The research goal.
            hypotheses: List of hypotheses.
            rankings: Ranking results.
            reflections: Reflection results.
            evolutions: Evolution results.
            iterations: Iteration history.
            
        Returns:
            The research overview.
        """
        system_message = """
        You are an expert scientific meta-reviewer tasked with synthesizing research results into a comprehensive overview.
        Your task is to:
        1. Summarize the research goal and process
        2. Identify key themes and patterns across hypotheses
        3. Highlight the most promising hypotheses and their strengths
        4. Outline critical limitations and areas for improvement
        5. Suggest future research directions
        
        Create a balanced, comprehensive overview that captures the current state of the research.
        """
        
        # Get the top-ranked hypotheses
        top_hypotheses = []
        if rankings:
            # Create a mapping of hypothesis IDs to hypotheses
            hypotheses_map = {h.get("id", ""): h for h in hypotheses}
            
            # Sort rankings by rank
            sorted_rankings = sorted(rankings, key=lambda x: x.get("rank", 999))
            
            # Get top 3 hypotheses or fewer if there are less
            for i, ranking in enumerate(sorted_rankings):
                if i >= 3:
                    break
                    
                h_id = ranking.get("id", "")
                if h_id in hypotheses_map:
                    h = hypotheses_map[h_id]
                    h_with_rank = {
                        "id": h_id,
                        "rank": ranking.get("rank", i + 1),
                        "content": h.get("content", {})
                    }
                    top_hypotheses.append(h_with_rank)
        else:
            # If no rankings, just use the first 3 hypotheses or fewer
            for i, h in enumerate(hypotheses):
                if i >= 3:
                    break
                
                top_hypotheses.append({
                    "id": h.get("id", ""),
                    "content": h.get("content", {})
                })
        
        # Count iterations
        iteration_count = len(iterations)
        
        # Prepare a summary of evolution patterns if available
        evolution_summary = "No evolution data available."
        if evolutions:
            evolution_types = {}
            for e in evolutions:
                e_type = e.get("evolution_type", "unknown")
                if e_type in evolution_types:
                    evolution_types[e_type] += 1
                else:
                    evolution_types[e_type] = 1
            
            evolution_summary = f"Evolution summary: {json.dumps(evolution_types)}"
        
        # Get key reflection points
        reflection_points = []
        for reflection in reflections:
            content = reflection.get("content", {})
            if isinstance(content, dict):
                reflection_type = reflection.get("reflection_type", "unknown")
                
                if reflection_type == "full_review":
                    strengths = content.get("strengths", [])
                    limitations = content.get("limitations", [])
                    
                    if strengths:
                        reflection_points.extend([f"Strength: {s}" for s in strengths[:2]])
                    
                    if limitations:
                        reflection_points.extend([f"Limitation: {l}" for l in limitations[:2]])
                
                elif reflection_type == "deep_verification":
                    assumptions = content.get("key_assumptions", [])
                    contradictions = content.get("potential_contradictions", [])
                    
                    if assumptions:
                        for a in assumptions[:2]:
                            if isinstance(a, dict):
                                assumption_text = a.get("assumption", "")
                                if assumption_text:
                                    reflection_points.append(f"Assumption: {assumption_text}")
                    
                    if contradictions:
                        reflection_points.extend([f"Contradiction: {c}" for c in contradictions[:2]])
        
        prompt = f"""
        # Research Goal
        {research_goal}
        
        # Research Process Summary
        - Number of iterations: {iteration_count}
        - Number of hypotheses: {len(hypotheses)}
        - {evolution_summary}
        
        # Top Hypotheses
        {json.dumps(top_hypotheses, indent=2)}
        
        # Key Reflection Points
        {json.dumps(reflection_points, indent=2)}
        
        Please synthesize this information into a comprehensive research overview.
        
        Format your response as a JSON object with the following structure:
        {{
            "research_goal_summary": "Concise summary of the research goal",
            "process_summary": "Summary of the research process and progress",
            "key_themes": ["List of key themes and patterns across hypotheses"],
            "top_hypotheses_summary": "Summary of the most promising hypotheses and their strengths",
            "critical_limitations": ["List of critical limitations and areas for improvement"],
            "future_directions": ["Suggested future research directions"],
            "overall_assessment": "Overall assessment of the current state of the research"
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
                    logger.error(f"Failed to parse research overview JSON: {json_str}")
                    return {"error": "Failed to parse research overview", "raw_response": response}
            else:
                logger.error(f"No JSON content found in response: {response}")
                return {"error": "No JSON content found", "raw_response": response}
            
        except Exception as e:
            logger.error(f"Error generating research overview: {e}")
            return {"error": str(e)} 