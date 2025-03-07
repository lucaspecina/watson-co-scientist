"""
Ranking Agent for conducting tournaments to rank hypotheses.
"""

import json
import logging
import random
from typing import Dict, List, Any, Optional, Tuple, Set

from .base_agent import BaseAgent
from ..config.config import SystemConfig
from ..core.models import Hypothesis, TournamentMatch, ResearchGoal

logger = logging.getLogger("co_scientist")

class RankingAgent(BaseAgent):
    """
    Agent responsible for conducting tournaments to rank hypotheses.
    """
    
    def __init__(self, config: SystemConfig):
        """
        Initialize the Ranking agent.
        
        Args:
            config (SystemConfig): The system configuration.
        """
        super().__init__("ranking", config)
    
    async def conduct_match(self, 
                       hypothesis1: Hypothesis, 
                       hypothesis2: Hypothesis, 
                       research_goal: ResearchGoal,
                       detailed: bool = False) -> TournamentMatch:
        """
        Conduct a tournament match between two hypotheses.
        
        Args:
            hypothesis1 (Hypothesis): The first hypothesis.
            hypothesis2 (Hypothesis): The second hypothesis.
            research_goal (ResearchGoal): The research goal.
            detailed (bool): Whether to conduct a detailed debate.
            
        Returns:
            TournamentMatch: The tournament match result.
        """
        logger.info(f"Conducting {'detailed' if detailed else 'simple'} match between hypotheses {hypothesis1.id} and {hypothesis2.id}")
        
        # Randomize the order of hypotheses to prevent bias
        if random.random() > 0.5:
            hypothesis1, hypothesis2 = hypothesis2, hypothesis1
        
        if detailed:
            # Build the prompt for a detailed scientific debate
            prompt = f"""
            You are judging a scientific debate between two competing hypotheses for the following research goal:
            
            Research Goal:
            {research_goal.text}
            
            Hypothesis A:
            Title: {hypothesis1.title}
            Summary: {hypothesis1.summary}
            Description: {hypothesis1.description}
            Supporting Evidence: {', '.join(hypothesis1.supporting_evidence)}
            
            Hypothesis B:
            Title: {hypothesis2.title}
            Summary: {hypothesis2.summary}
            Description: {hypothesis2.description}
            Supporting Evidence: {', '.join(hypothesis2.supporting_evidence)}
            
            Conduct a detailed scientific debate between these hypotheses:
            
            Round 1: Present the key strengths and weaknesses of each hypothesis.
            Round 2: Have each hypothesis respond to the criticisms and highlight advantages over the competitor.
            Round 3: Final arguments synthesizing the debate and addressing key points.
            
            After the debate, determine which hypothesis is superior based on:
            1. Scientific validity and correctness
            2. Novelty and originality
            3. Testability and falsifiability
            4. Alignment with the research goal
            5. Overall quality and potential impact
            
            Format your response as a JSON object with the following structure:
            
            ```json
            {{
                "debate_transcript": "Full transcript of the three-round scientific debate...",
                "evaluation": {{
                    "hypothesis_a": {{
                        "strengths": ["Strength 1", "Strength 2", ...],
                        "weaknesses": ["Weakness 1", "Weakness 2", ...]
                    }},
                    "hypothesis_b": {{
                        "strengths": ["Strength 1", "Strength 2", ...],
                        "weaknesses": ["Weakness 1", "Weakness 2", ...]
                    }}
                }},
                "winner": "A", "B", or "tie",
                "rationale": "Detailed rationale for the decision..."
            }}
            ```
            
            Be fair, balanced, and objective in your evaluation.
            """
        else:
            # Build the prompt for a simple comparison
            prompt = f"""
            You are comparing two competing hypotheses for the following research goal:
            
            Research Goal:
            {research_goal.text}
            
            Hypothesis A:
            Title: {hypothesis1.title}
            Summary: {hypothesis1.summary}
            Description: {hypothesis1.description}
            Supporting Evidence: {', '.join(hypothesis1.supporting_evidence)}
            
            Hypothesis B:
            Title: {hypothesis2.title}
            Summary: {hypothesis2.summary}
            Description: {hypothesis2.description}
            Supporting Evidence: {', '.join(hypothesis2.supporting_evidence)}
            
            Determine which hypothesis is superior based on:
            1. Scientific validity and correctness
            2. Novelty and originality
            3. Testability and falsifiability
            4. Alignment with the research goal
            5. Overall quality and potential impact
            
            Format your response as a JSON object with the following structure:
            
            ```json
            {{
                "evaluation": {{
                    "hypothesis_a": {{
                        "strengths": ["Strength 1", "Strength 2", ...],
                        "weaknesses": ["Weakness 1", "Weakness 2", ...]
                    }},
                    "hypothesis_b": {{
                        "strengths": ["Strength 1", "Strength 2", ...],
                        "weaknesses": ["Weakness 1", "Weakness 2", ...]
                    }}
                }},
                "winner": "A", "B", or "tie",
                "rationale": "Detailed rationale for the decision..."
            }}
            ```
            
            Be fair, balanced, and objective in your evaluation.
            """
        
        # Generate match result
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
            
            # Determine winner
            winner_id = None
            if data["winner"] == "A":
                winner_id = hypothesis1.id
            elif data["winner"] == "B":
                winner_id = hypothesis2.id
            # For "tie", winner_id remains None
            
            # Build the debate transcript if not provided
            if "debate_transcript" not in data:
                # Create a simple debate transcript from the evaluation
                data["debate_transcript"] = f"""
                # Comparison of Hypotheses
                
                ## Hypothesis A: {hypothesis1.title}
                
                ### Strengths:
                {chr(10).join(['- ' + s for s in data["evaluation"]["hypothesis_a"]["strengths"]])}
                
                ### Weaknesses:
                {chr(10).join(['- ' + w for w in data["evaluation"]["hypothesis_a"]["weaknesses"]])}
                
                ## Hypothesis B: {hypothesis2.title}
                
                ### Strengths:
                {chr(10).join(['- ' + s for s in data["evaluation"]["hypothesis_b"]["strengths"]])}
                
                ### Weaknesses:
                {chr(10).join(['- ' + w for w in data["evaluation"]["hypothesis_b"]["weaknesses"]])}
                
                ## Decision: {data["winner"] if data["winner"] in ["A", "B"] else "Tie"}
                
                ## Rationale:
                {data["rationale"]}
                """
            
            # Create the tournament match
            match = TournamentMatch(
                hypothesis1_id=hypothesis1.id,
                hypothesis2_id=hypothesis2.id,
                winner_id=winner_id,
                rationale=data["rationale"],
                debate_transcript=data["debate_transcript"],
                judge="ranking"
            )
            
            logger.info(f"Match completed with winner: {'Tie' if winner_id is None else winner_id}")
            return match
            
        except Exception as e:
            logger.error(f"Error parsing match result from response: {e}")
            logger.debug(f"Response: {response}")
            
            # Create a basic match in case of parsing error
            match = TournamentMatch(
                hypothesis1_id=hypothesis1.id,
                hypothesis2_id=hypothesis2.id,
                winner_id=None,  # Tie in case of error
                rationale=f"Error parsing match result: {str(e)}",
                debate_transcript=response,
                judge="ranking"
            )
            
            return match
    
    def update_elo_ratings(self, 
                          match: TournamentMatch, 
                          hypothesis1: Hypothesis, 
                          hypothesis2: Hypothesis,
                          k_factor: float = 32.0) -> Tuple[Hypothesis, Hypothesis]:
        """
        Update Elo ratings based on the match result.
        
        Args:
            match (TournamentMatch): The tournament match.
            hypothesis1 (Hypothesis): The first hypothesis.
            hypothesis2 (Hypothesis): The second hypothesis.
            k_factor (float): The K-factor for Elo calculation.
            
        Returns:
            Tuple[Hypothesis, Hypothesis]: The updated hypotheses.
        """
        # Get current ratings
        r1 = hypothesis1.elo_rating
        r2 = hypothesis2.elo_rating
        
        # Calculate expected scores
        e1 = 1 / (1 + 10 ** ((r2 - r1) / 400))
        e2 = 1 / (1 + 10 ** ((r1 - r2) / 400))
        
        # Calculate actual scores
        if match.winner_id == hypothesis1.id:
            s1, s2 = 1.0, 0.0
        elif match.winner_id == hypothesis2.id:
            s1, s2 = 0.0, 1.0
        else:
            s1, s2 = 0.5, 0.5  # Tie
        
        # Update ratings
        hypothesis1.elo_rating = r1 + k_factor * (s1 - e1)
        hypothesis2.elo_rating = r2 + k_factor * (s2 - e2)
        
        # Update matches played
        hypothesis1.matches_played += 1
        hypothesis2.matches_played += 1
        
        logger.info(f"Updated Elo ratings: {hypothesis1.id} = {hypothesis1.elo_rating:.2f}, {hypothesis2.id} = {hypothesis2.elo_rating:.2f}")
        
        return hypothesis1, hypothesis2
    
    def select_pairs_for_tournament(self, 
                                  hypotheses: List[Hypothesis], 
                                  matches_played: Dict[Tuple[str, str], bool],
                                  similarity_groups: Optional[Dict[str, Set[str]]] = None,
                                  num_pairs: int = 10) -> List[Tuple[Hypothesis, Hypothesis]]:
        """
        Select pairs of hypotheses for tournament matches.
        
        Args:
            hypotheses (List[Hypothesis]): The list of hypotheses.
            matches_played (Dict[Tuple[str, str], bool]): Dictionary indicating which pairs have played.
            similarity_groups (Optional[Dict[str, Set[str]]]): Groups of similar hypotheses.
            num_pairs (int): The number of pairs to select.
            
        Returns:
            List[Tuple[Hypothesis, Hypothesis]]: The selected pairs.
        """
        # Sort hypotheses by Elo rating and matches played
        sorted_hypotheses = sorted(
            hypotheses, 
            key=lambda h: (h.elo_rating, -h.matches_played), 
            reverse=True
        )
        
        pairs = []
        
        # Helper function to check if a pair has played
        def has_played(h1: Hypothesis, h2: Hypothesis) -> bool:
            key1 = (h1.id, h2.id)
            key2 = (h2.id, h1.id)
            return matches_played.get(key1, False) or matches_played.get(key2, False)
        
        # First, try to match hypotheses with similar ones
        if similarity_groups:
            for h1 in sorted_hypotheses:
                if len(pairs) >= num_pairs:
                    break
                    
                # Get similar hypotheses
                similar_ids = similarity_groups.get(h1.id, set())
                
                # Find a similar hypothesis that hasn't played against h1
                for h2 in sorted_hypotheses:
                    if h1.id != h2.id and h2.id in similar_ids and not has_played(h1, h2):
                        pairs.append((h1, h2))
                        matches_played[(h1.id, h2.id)] = True
                        break
        
        # Fill remaining slots with top hypotheses vs others
        for h1 in sorted_hypotheses:
            if len(pairs) >= num_pairs:
                break
                
            for h2 in sorted_hypotheses:
                if h1.id != h2.id and not has_played(h1, h2):
                    pairs.append((h1, h2))
                    matches_played[(h1.id, h2.id)] = True
                    break
        
        logger.info(f"Selected {len(pairs)} pairs for tournament matches")
        return pairs