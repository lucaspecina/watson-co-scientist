"""
Coordinator for mini-RAUL.
This module orchestrates the multi-agent system.
"""
import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple, Union

from .session import Session, SessionState
from ..agents.generation_agent import GenerationAgent
from ..agents.ranking_agent import RankingAgent
from ..agents.reflection_agent import ReflectionAgent
from ..agents.evolution_agent import EvolutionAgent
from ..agents.meta_review_agent import MetaReviewAgent

logger = logging.getLogger(__name__)


class Coordinator:
    """
    Coordinator for the multi-agent research system.
    Orchestrates the different agents and manages the research process.
    """
    
    def __init__(self, session: Session, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the coordinator.
        
        Args:
            session: The research session to coordinate.
            config: Optional configuration for the coordinator.
        """
        self.session = session
        self.config = config or {}
        
        # Initialize agents
        self._init_agents()
    
    def _init_agents(self) -> None:
        """Initialize all agent types."""
        self.generation_agents = []
        self.ranking_agents = []
        self.reflection_agents = []
        self.evolution_agents = []
        self.meta_agents = []
        
        # Generation agents
        generation_configs = self.config.get("generation_agents", [
            {"name": "literature_explorer", "config": {"generation_type": "literature_exploration"}},
            {"name": "debate_simulator", "config": {"generation_type": "scientific_debate"}}
        ])
        
        for cfg in generation_configs:
            self.generation_agents.append(GenerationAgent(**cfg))
        
        # Ranking agents
        ranking_configs = self.config.get("ranking_agents", [
            {"name": "tournament_ranker"}
        ])
        
        for cfg in ranking_configs:
            self.ranking_agents.append(RankingAgent(**cfg))
        
        # Reflection agents
        reflection_configs = self.config.get("reflection_agents", [
            {"name": "full_reviewer", "config": {"reflection_type": "full_review"}},
            {"name": "tournament_reviewer", "config": {"reflection_type": "tournament_review"}},
            {"name": "deep_verifier", "config": {"reflection_type": "deep_verification"}}
        ])
        
        for cfg in reflection_configs:
            self.reflection_agents.append(ReflectionAgent(**cfg))
        
        # Evolution agents
        evolution_configs = self.config.get("evolution_agents", [
            {"name": "improver", "config": {"evolution_type": "improvement"}},
            {"name": "simplifier", "config": {"evolution_type": "simplification"}},
            {"name": "extender", "config": {"evolution_type": "extension"}}
        ])
        
        for cfg in evolution_configs:
            self.evolution_agents.append(EvolutionAgent(**cfg))
        
        # Meta-review agents
        meta_configs = self.config.get("meta_agents", [
            {"name": "meta_reviewer"}
        ])
        
        for cfg in meta_configs:
            self.meta_agents.append(MetaReviewAgent(**cfg))
    
    async def start_session(self, research_goal: str, preferences: str = "", constraints: str = "") -> None:
        """
        Start a new research session.
        
        Args:
            research_goal: The research goal.
            preferences: User preferences.
            constraints: User constraints.
        """
        # Set the research goal
        self.session.research_goal = research_goal
        
        # Add preferences and constraints as metadata
        self.session.add_metadata("preferences", preferences)
        self.session.add_metadata("constraints", constraints)
        
        # Start the first iteration
        self.session.start_iteration()
        self.session.state = SessionState.PLANNING
        
        logger.info(f"Started session {self.session.session_id} with research goal: {research_goal}")
        
        # Continue with the first iteration
        await self.run_iteration()
    
    async def run_iteration(self) -> None:
        """Run a complete iteration of the research process."""
        if not self.session.current_iteration:
            raise ValueError("No active iteration")
        
        # Create the context for this iteration
        context = self._create_iteration_context()
        
        # === GENERATION PHASE ===
        self.session.state = SessionState.GENERATING
        await self._run_generation_phase(context)
        
        # === RANKING PHASE ===
        if self.session.current_iteration["hypotheses"]:
            self.session.state = SessionState.RANKING
            await self._run_ranking_phase(context)
        
        # === REFLECTION PHASE ===
        if self.session.current_iteration["hypotheses"]:
            self.session.state = SessionState.REFLECTING
            await self._run_reflection_phase(context)
        
        # === EVOLUTION PHASE ===
        if self.session.current_iteration["hypotheses"] and self.session.current_iteration["reflections"]:
            self.session.state = SessionState.EVOLVING
            await self._run_evolution_phase(context)
        
        # === WAITING FOR FEEDBACK ===
        self.session.state = SessionState.WAITING_FEEDBACK
        logger.info(f"Iteration {self.session.current_iteration['iteration_number']} completed, waiting for user feedback")
    
    async def add_user_feedback(self, feedback: str) -> None:
        """
        Add user feedback to the current iteration.
        
        Args:
            feedback: The user feedback.
        """
        self.session.add_user_feedback(feedback)
        logger.info(f"Added user feedback to iteration {self.session.current_iteration['iteration_number']}")
    
    async def create_meta_review(self) -> Dict[str, Any]:
        """
        Create a meta-review of the research progress.
        
        Returns:
            The meta-review.
        """
        if not self.meta_agents:
            raise ValueError("No meta-review agents available")
        
        # Create the context for meta-review
        context = self._create_meta_review_context()
        
        # Run the first meta-review agent
        meta_agent = self.meta_agents[0]
        meta_review = await meta_agent.run(context)
        
        return meta_review
    
    async def _run_generation_phase(self, context: Dict[str, Any]) -> None:
        """
        Run the generation phase.
        
        Args:
            context: The context for this iteration.
        """
        logger.info("Starting generation phase")
        
        # Run all generation agents in parallel
        tasks = []
        for agent in self.generation_agents:
            tasks.append(agent.run(context))
        
        # Wait for all generation agents to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process the results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error in generation agent {self.generation_agents[i].name}: {result}")
                continue
            
            # Add the hypothesis to the session
            self.session.add_hypothesis(result, self.generation_agents[i].name)
            
        logger.info(f"Generated {len(self.session.current_iteration['hypotheses'])} hypotheses")
    
    async def _run_ranking_phase(self, context: Dict[str, Any]) -> None:
        """
        Run the ranking phase.
        
        Args:
            context: The context for this iteration.
        """
        logger.info("Starting ranking phase")
        
        # Only run ranking if we have at least 2 hypotheses
        if len(self.session.current_iteration["hypotheses"]) < 2:
            logger.warning("Not enough hypotheses to rank")
            return
        
        # Add hypotheses to the context
        ranking_context = {
            **context,
            "hypotheses": self.session.current_iteration["hypotheses"]
        }
        
        # Run the first ranking agent
        if self.ranking_agents:
            try:
                result = await self.ranking_agents[0].run(ranking_context)
                
                # Add the ranking to the session
                self.session.add_ranking(result["rankings"], self.ranking_agents[0].name)
                
                logger.info(f"Ranked {len(result['rankings'])} hypotheses")
            except Exception as e:
                logger.error(f"Error in ranking agent {self.ranking_agents[0].name}: {e}")
    
    async def _run_reflection_phase(self, context: Dict[str, Any]) -> None:
        """
        Run the reflection phase.
        
        Args:
            context: The context for this iteration.
        """
        logger.info("Starting reflection phase")
        
        # Run a full review for each hypothesis
        full_reviewer = next((a for a in self.reflection_agents if a.reflection_type == "full_review"), None)
        
        if full_reviewer:
            for hypothesis in self.session.current_iteration["hypotheses"]:
                reflection_context = {
                    **context,
                    "hypothesis": hypothesis
                }
                
                try:
                    result = await full_reviewer.run(reflection_context)
                    
                    # Add the reflection to the session
                    self.session.add_reflection(result, full_reviewer.name)
                    
                    logger.info(f"Created full review for hypothesis {hypothesis.get('id', '')}")
                except Exception as e:
                    logger.error(f"Error in reflection agent {full_reviewer.name}: {e}")
        
        # Run a tournament review if we have rankings
        if self.session.current_iteration["rankings"]:
            tournament_reviewer = next((a for a in self.reflection_agents if a.reflection_type == "tournament_review"), None)
            
            if tournament_reviewer:
                tournament_context = {
                    **context,
                    "hypotheses": self.session.current_iteration["hypotheses"],
                    "rankings": self.session.current_iteration["rankings"][0]["rankings"]  # Use the first ranking
                }
                
                try:
                    result = await tournament_reviewer.run(tournament_context)
                    
                    # Add the reflection to the session
                    self.session.add_reflection(result, tournament_reviewer.name)
                    
                    logger.info("Created tournament review")
                except Exception as e:
                    logger.error(f"Error in reflection agent {tournament_reviewer.name}: {e}")
        
        # Run a deep verification for the top-ranked hypothesis
        if self.session.current_iteration["rankings"] and self.session.current_iteration["hypotheses"]:
            deep_verifier = next((a for a in self.reflection_agents if a.reflection_type == "deep_verification"), None)
            
            if deep_verifier:
                # Find the top-ranked hypothesis
                rankings = self.session.current_iteration["rankings"][0]["rankings"]  # Use the first ranking
                top_rank = min([r.get("rank", 999) for r in rankings])
                top_id = next((r.get("id", "") for r in rankings if r.get("rank", 999) == top_rank), None)
                
                if top_id:
                    top_hypothesis = next((h for h in self.session.current_iteration["hypotheses"] if h.get("id", "") == top_id), None)
                    
                    if top_hypothesis:
                        verification_context = {
                            **context,
                            "hypothesis": top_hypothesis
                        }
                        
                        try:
                            result = await deep_verifier.run(verification_context)
                            
                            # Add the reflection to the session
                            self.session.add_reflection(result, deep_verifier.name)
                            
                            logger.info(f"Created deep verification for top hypothesis {top_id}")
                        except Exception as e:
                            logger.error(f"Error in reflection agent {deep_verifier.name}: {e}")
    
    async def _run_evolution_phase(self, context: Dict[str, Any]) -> None:
        """
        Run the evolution phase.
        
        Args:
            context: The context for this iteration.
        """
        logger.info("Starting evolution phase")
        
        # Get the top-ranked hypothesis
        top_hypothesis = None
        
        if self.session.current_iteration["rankings"]:
            rankings = self.session.current_iteration["rankings"][0]["rankings"]  # Use the first ranking
            top_rank = min([r.get("rank", 999) for r in rankings])
            top_id = next((r.get("id", "") for r in rankings if r.get("rank", 999) == top_rank), None)
            
            if top_id:
                top_hypothesis = next((h for h in self.session.current_iteration["hypotheses"] if h.get("id", "") == top_id), None)
        
        # If no ranking, use the first hypothesis
        if not top_hypothesis and self.session.current_iteration["hypotheses"]:
            top_hypothesis = self.session.current_iteration["hypotheses"][0]
        
        if not top_hypothesis:
            logger.warning("No hypothesis available for evolution")
            return
        
        # Get the reflection for this hypothesis
        hypothesis_reflections = []
        
        for reflection in self.session.current_iteration["reflections"]:
            # Check if this reflection is about the top hypothesis
            content = reflection.get("content", {})
            if isinstance(content, dict) and content.get("reflection", {}).get("hypothesis", {}).get("id", "") == top_hypothesis.get("id", ""):
                hypothesis_reflections.append(content)
        
        # Run each evolution agent
        for agent in self.evolution_agents:
            evolution_context = {
                **context,
                "hypothesis": top_hypothesis,
                "reflection": hypothesis_reflections[0] if hypothesis_reflections else {},
                "user_feedback": self.session.current_iteration["user_feedback"]
            }
            
            try:
                result = await agent.run(evolution_context)
                
                # Add the evolution to the session
                self.session.add_evolution(result, agent.name)
                
                logger.info(f"Created {agent.evolution_type} evolution for hypothesis {top_hypothesis.get('id', '')}")
            except Exception as e:
                logger.error(f"Error in evolution agent {agent.name}: {e}")
    
    def _create_iteration_context(self) -> Dict[str, Any]:
        """
        Create the context for the current iteration.
        
        Returns:
            The iteration context.
        """
        return {
            "research_goal": self.session.research_goal,
            "preferences": self.session.data["metadata"].get("preferences", ""),
            "constraints": self.session.data["metadata"].get("constraints", ""),
            "iteration": self.session.current_iteration["iteration_number"],
            "previous_iterations": self.session.data["iterations"][:-1]  # All iterations except the current one
        }
    
    def _create_meta_review_context(self) -> Dict[str, Any]:
        """
        Create the context for meta-review.
        
        Returns:
            The meta-review context.
        """
        # Collect all hypotheses, rankings, reflections, and evolutions from all iterations
        all_hypotheses = []
        all_rankings = []
        all_reflections = []
        all_evolutions = []
        
        for iteration in self.session.data["iterations"]:
            all_hypotheses.extend(iteration["hypotheses"])
            all_rankings.extend(iteration["rankings"])
            all_reflections.extend(iteration["reflections"])
            all_evolutions.extend(iteration["evolution"])
        
        return {
            "research_goal": self.session.research_goal,
            "preferences": self.session.data["metadata"].get("preferences", ""),
            "constraints": self.session.data["metadata"].get("constraints", ""),
            "hypotheses": all_hypotheses,
            "rankings": all_rankings,
            "reflections": all_reflections,
            "evolutions": all_evolutions,
            "iterations": self.session.data["iterations"]
        } 