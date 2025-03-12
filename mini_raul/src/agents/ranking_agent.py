"""
Ranking agent for mini-RAUL.
This agent compares and ranks research hypotheses.
"""
import logging
import json
from typing import Dict, Any, List, Optional, Tuple

from ..core.agent import Agent
from ..models.llm import LLMProvider

logger = logging.getLogger(__name__)


class RankingAgent(Agent):
    """
    Ranking agent that compares and ranks research hypotheses.
    Implements tournament-style ranking with scientific debate.
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ranking agent.
        
        Args:
            name: The name of the agent.
            config: Optional configuration for the agent.
        """
        super().__init__(name, config)
        
        # Initialize LLM client
        llm_provider = self.config.get("llm_provider", "azure")
        llm_config = self.config.get("llm_config", {})
        self.llm = LLMProvider.get_client(llm_provider, llm_config)
        
        self.temperature = self.config.get("temperature", 0.4)  # Lower temperature for more consistent ranking
        
    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Rank a set of research hypotheses.
        
        Args:
            context: The context information including hypotheses to rank.
            
        Returns:
            Dict containing the ranking results.
        """
        research_goal = context.get("research_goal", "")
        hypotheses = context.get("hypotheses", [])
        
        if not research_goal:
            raise ValueError("Research goal is required")
        
        if not hypotheses or len(hypotheses) < 2:
            logger.warning("Not enough hypotheses to rank")
            return {
                "rankings": [{"id": h.get("id"), "rank": 1} for h in hypotheses],
                "debate_summaries": [],
                "agent_name": self.name
            }
        
        # Organize hypotheses into pairs for tournament-style ranking
        pairs = self._create_hypothesis_pairs(hypotheses)
        
        # Run debates for each pair
        debate_results = []
        
        for pair in pairs:
            result = await self._debate_hypotheses(
                research_goal=research_goal,
                hypothesis1=pair[0],
                hypothesis2=pair[1]
            )
            debate_results.append(result)
        
        # Calculate final rankings based on wins/losses
        rankings = self._calculate_rankings(hypotheses, debate_results)
        
        return {
            "rankings": rankings,
            "debate_summaries": debate_results,
            "agent_name": self.name
        }
    
    def _create_hypothesis_pairs(self, hypotheses: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """
        Create pairs of hypotheses for tournament-style ranking.
        
        Args:
            hypotheses: List of hypotheses to pair.
            
        Returns:
            List of hypothesis pairs.
        """
        pairs = []
        n = len(hypotheses)
        
        # Simple round-robin pairing
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((hypotheses[i], hypotheses[j]))
        
        return pairs
    
    async def _debate_hypotheses(self, 
                               research_goal: str,
                               hypothesis1: Dict[str, Any],
                               hypothesis2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Conduct a debate between two hypotheses.
        
        Args:
            research_goal: The research goal.
            hypothesis1: The first hypothesis.
            hypothesis2: The second hypothesis.
            
        Returns:
            The debate result.
        """
        system_message = """
        You are an expert scientific reviewer comparing two research hypotheses for the same research goal.
        You will thoroughly evaluate each hypothesis based on:
        1. Scientific validity and plausibility
        2. Novelty and originality
        3. Potential impact if proven correct
        4. Experimental feasibility and testability
        5. Alignment with the research goal
        
        After evaluating both hypotheses, you will:
        1. Provide a detailed comparison identifying strengths and weaknesses of each
        2. Select the superior hypothesis (either Hypothesis 1 or Hypothesis 2)
        3. Explain your reasoning for the selection
        
        Be fair, thorough, and objective in your evaluation.
        """
        
        # Get the hypothesis content
        h1_content = hypothesis1.get("content", {}).get("hypothesis", {})
        h2_content = hypothesis2.get("content", {}).get("hypothesis", {})
        
        h1_id = hypothesis1.get("id", "h1")
        h2_id = hypothesis2.get("id", "h2")
        
        # Convert to strings if needed
        if isinstance(h1_content, dict):
            h1_str = json.dumps(h1_content, indent=2)
        else:
            h1_str = str(h1_content)
            
        if isinstance(h2_content, dict):
            h2_str = json.dumps(h2_content, indent=2)
        else:
            h2_str = str(h2_content)
        
        prompt = f"""
        # Research Goal
        {research_goal}
        
        # Hypothesis 1
        {h1_str}
        
        # Hypothesis 2
        {h2_str}
        
        Please evaluate these two hypotheses thoroughly and determine which one is superior.
        
        Format your response as a JSON object with the following structure:
        {{
            "comparison": "Detailed comparison of the two hypotheses, including strengths and weaknesses of each",
            "winner": "Either 'Hypothesis 1' or 'Hypothesis 2'",
            "reasoning": "Detailed explanation of why the winner was selected"
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
                    
                    # Add hypothesis IDs for reference
                    debate_result = {
                        "hypothesis1_id": h1_id,
                        "hypothesis2_id": h2_id,
                        "comparison": result.get("comparison", ""),
                        "winner": None,
                        "reasoning": result.get("reasoning", "")
                    }
                    
                    # Determine winner by ID
                    winner_text = result.get("winner", "").strip().lower()
                    if "hypothesis 1" in winner_text:
                        debate_result["winner"] = h1_id
                    elif "hypothesis 2" in winner_text:
                        debate_result["winner"] = h2_id
                    
                    return debate_result
                    
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse debate JSON: {json_str}")
                    return {
                        "hypothesis1_id": h1_id,
                        "hypothesis2_id": h2_id,
                        "error": "Failed to parse debate result",
                        "raw_response": response
                    }
            else:
                logger.error(f"No JSON content found in response: {response}")
                return {
                    "hypothesis1_id": h1_id,
                    "hypothesis2_id": h2_id,
                    "error": "No JSON content found",
                    "raw_response": response
                }
            
        except Exception as e:
            logger.error(f"Error during hypothesis debate: {e}")
            return {
                "hypothesis1_id": h1_id,
                "hypothesis2_id": h2_id,
                "error": str(e)
            }
    
    def _calculate_rankings(self, 
                           hypotheses: List[Dict[str, Any]], 
                           debate_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Calculate rankings based on debate results.
        
        Args:
            hypotheses: The original hypotheses.
            debate_results: Results of all debates.
            
        Returns:
            List of rankings.
        """
        # Create a win-loss record for each hypothesis
        records = {}
        
        # Initialize records
        for h in hypotheses:
            h_id = h.get("id", "")
            if h_id:
                records[h_id] = {"id": h_id, "wins": 0, "losses": 0}
        
        # Count wins and losses
        for result in debate_results:
            winner = result.get("winner")
            h1_id = result.get("hypothesis1_id")
            h2_id = result.get("hypothesis2_id")
            
            if winner and h1_id in records and h2_id in records:
                if winner == h1_id:
                    records[h1_id]["wins"] += 1
                    records[h2_id]["losses"] += 1
                elif winner == h2_id:
                    records[h2_id]["wins"] += 1
                    records[h1_id]["losses"] += 1
        
        # Convert to list and sort by wins (descending)
        record_list = list(records.values())
        record_list.sort(key=lambda x: x["wins"], reverse=True)
        
        # Assign ranks (tie handling: same win count = same rank)
        current_rank = 1
        current_wins = -1
        
        rankings = []
        
        for i, record in enumerate(record_list):
            if record["wins"] != current_wins:
                current_rank = i + 1
                current_wins = record["wins"]
            
            rankings.append({
                "id": record["id"],
                "rank": current_rank,
                "wins": record["wins"],
                "losses": record["losses"]
            })
        
        return rankings 