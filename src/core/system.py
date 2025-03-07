"""
Core system implementation for the Co-Scientist system.
Coordinates agents and manages the overall workflow.
"""

import os
import json
import logging
import asyncio
import random
from typing import Dict, List, Any, Optional, Set, Tuple, Type
from datetime import datetime

from .database import Database
from .models import (
    ResearchGoal, 
    Hypothesis, 
    ExperimentalProtocol, 
    Review, 
    TournamentMatch, 
    ResearchOverview,
    MetaReview,
    HypothesisStatus,
    HypothesisSource
)
from ..config.config import load_config, SystemConfig
from ..agents.supervisor_agent import SupervisorAgent
from ..agents.generation_agent import GenerationAgent
from ..agents.reflection_agent import ReflectionAgent
from ..agents.ranking_agent import RankingAgent
from ..agents.proximity_agent import ProximityAgent
from ..agents.evolution_agent import EvolutionAgent
from ..agents.meta_review_agent import MetaReviewAgent

logger = logging.getLogger("co_scientist")

class CoScientistSystem:
    """Main system controller for the Co-Scientist system."""
    
    def __init__(self, config_name: str = "default", data_dir: str = "data"):
        """
        Initialize the Co-Scientist system.
        
        Args:
            config_name (str): The name of the configuration to use.
            data_dir (str): The directory to store data in.
        """
        # Load system configuration
        self.config = load_config(config_name)
        
        # Initialize database
        self.db = Database(data_dir)
        
        # Store a reference to this system in the config for agent access
        self.config.system = self
        
        # Initialize agents
        self.supervisor = SupervisorAgent(self.config)
        self.generation = GenerationAgent(self.config)
        self.reflection = ReflectionAgent(self.config)
        self.ranking = RankingAgent(self.config)
        self.proximity = ProximityAgent(self.config)
        self.evolution = EvolutionAgent(self.config)
        self.meta_review = MetaReviewAgent(self.config)
        
        # Initialize state
        self.current_research_goal: Optional[ResearchGoal] = None
        self.research_plan_config: Dict[str, Any] = {}
        self.agent_weights: Dict[str, float] = {}
        self.current_state: Dict[str, Any] = {}
        self.similarity_groups: Dict[str, Set[str]] = {}
        self.matches_played: Dict[Tuple[str, str], bool] = {}
        
        logger.info(f"Initialized Co-Scientist system with configuration '{config_name}'")
    
    async def analyze_research_goal(self, research_goal_text: str) -> ResearchGoal:
        """
        Analyze a research goal and initialize the system for processing it.
        
        Args:
            research_goal_text (str): The research goal text.
            
        Returns:
            ResearchGoal: The created research goal.
        """
        logger.info(f"Analyzing research goal: {research_goal_text[:100]}...")
        
        # Create the research goal
        self.current_research_goal = ResearchGoal(
            text=research_goal_text,
            preferences={},
            constraints={}
        )
        
        # Save to database
        self.db.research_goals.save(self.current_research_goal)
        
        # Parse the research goal
        self.research_plan_config = await self.supervisor.parse_research_goal(research_goal_text)
        
        # Update the research goal with preferences and constraints
        self.current_research_goal.preferences = self.research_plan_config.get("preferences", {})
        self.current_research_goal.constraints = {"constraints": self.research_plan_config.get("constraints", [])}
        
        # Save the updated research goal
        self.db.research_goals.save(self.current_research_goal)
        
        # Initialize state
        self.current_state = {
            "num_hypotheses": 0,
            "num_reviews": 0,
            "num_tournament_matches": 0,
            "num_protocols": 0,
            "top_hypotheses": [],
            "iterations_completed": 0,
            "last_research_overview_id": None
        }
        
        # Initialize agent weights
        self.agent_weights = {
            "generation": 0.65,
            "reflection": 0.2,
            "ranking": 0.05,
            "proximity": 0.0,
            "evolution": 0.0,
            "meta_review": 0.05,
            "protocol_generation": 0.05
        }
        
        logger.info(f"Initialized system for research goal {self.current_research_goal.id}")
        return self.current_research_goal
    
    async def run_iteration(self) -> Dict[str, Any]:
        """
        Run one iteration of the system workflow.
        
        Returns:
            Dict[str, Any]: The updated state.
        """
        if not self.current_research_goal:
            raise ValueError("No research goal set. Call analyze_research_goal first.")
            
        logger.info(f"Starting iteration {self.current_state['iterations_completed'] + 1} for research goal {self.current_research_goal.id}")
        
        # Run agents based on weights
        tasks = []
        
        # Generation
        if self.agent_weights["generation"] > 0.05:
            num_to_generate = int(self.agent_weights["generation"] * 5)
            for _ in range(num_to_generate):
                tasks.append(self._generate_hypothesis())
        
        # Reflection
        if self.agent_weights["reflection"] > 0.05:
            unreviewed_hypotheses = self._get_unreviewed_hypotheses()
            num_to_review = min(len(unreviewed_hypotheses), int(self.agent_weights["reflection"] * 10))
            hypotheses_to_review = unreviewed_hypotheses[:num_to_review]
            for hypothesis in hypotheses_to_review:
                tasks.append(self._review_hypothesis(hypothesis))
                
            # Also review protocols if we have any
            unreviewed_protocols = self._get_unreviewed_protocols()
            num_protocols_to_review = min(len(unreviewed_protocols), int(self.agent_weights["reflection"] * 2))
            for protocol_data in unreviewed_protocols[:num_protocols_to_review]:
                tasks.append(self._review_protocol(protocol_data["protocol"], protocol_data["hypothesis"]))
        
        # Ranking
        if self.agent_weights["ranking"] > 0.05:
            tasks.append(self._run_tournament())
        
        # Proximity
        if self.agent_weights["proximity"] > 0.05:
            tasks.append(self._update_similarity_groups())
        
        # Evolution
        if self.agent_weights["evolution"] > 0.05:
            num_to_evolve = int(self.agent_weights["evolution"] * 3)
            for _ in range(num_to_evolve):
                tasks.append(self._evolve_hypothesis())
        
        # Protocol Generation
        if self.agent_weights["protocol_generation"] > 0.05:
            num_to_generate = int(self.agent_weights["protocol_generation"] * 3)
            for _ in range(num_to_generate):
                tasks.append(self._generate_protocol())
        
        # Meta-review
        if self.agent_weights["meta_review"] > 0.05:
            tasks.append(self._generate_meta_review())
            
            # Generate research overview periodically
            if self.current_state["iterations_completed"] % 3 == 0:
                tasks.append(self._generate_research_overview())
        
        # Run all tasks concurrently
        await asyncio.gather(*tasks)
        
        # Update state
        self.current_state["iterations_completed"] += 1
        
        # Update hypotheses count
        self.current_state["num_hypotheses"] = len(self.db.hypotheses.get_all())
        
        # Update reviews count
        self.current_state["num_reviews"] = len(self.db.reviews.get_all())
        
        # Update tournament matches count
        self.current_state["num_tournament_matches"] = len(self.db.tournament_matches.get_all())
        
        # Update protocols count
        self.current_state["num_protocols"] = len(self.db.experimental_protocols.get_all())
        
        # Update top hypotheses
        all_hypotheses = self.db.hypotheses.get_all()
        sorted_hypotheses = sorted(all_hypotheses, key=lambda h: h.elo_rating, reverse=True)
        self.current_state["top_hypotheses"] = [h.id for h in sorted_hypotheses[:10]]
        
        # Re-plan agent weights for next iteration
        self.agent_weights = await self.supervisor.plan_task_allocation(
            self.current_research_goal,
            self.research_plan_config,
            self.current_state
        )
        
        return self.current_state
    
    async def run_until_completion(self, max_iterations: int = 10) -> ResearchOverview:
        """
        Run the system until completion or until max_iterations is reached.
        
        Args:
            max_iterations (int): The maximum number of iterations to run.
            
        Returns:
            ResearchOverview: The final research overview.
        """
        if not self.current_research_goal:
            raise ValueError("No research goal set. Call analyze_research_goal first.")
            
        logger.info(f"Running system until completion (max {max_iterations} iterations)")
        
        for i in range(max_iterations):
            logger.info(f"Starting iteration {i+1}/{max_iterations}")
            
            # Run one iteration
            await self.run_iteration()
            
            # Check if we should terminate
            terminate, reason = await self.supervisor.evaluate_termination(
                self.current_research_goal,
                self.current_state,
                self.research_plan_config
            )
            
            if terminate:
                logger.info(f"Terminating after {i+1} iterations: {reason}")
                break
        
        # Generate a final research overview
        research_overview = await self._generate_research_overview()
        
        logger.info(f"System completed after {self.current_state['iterations_completed']} iterations")
        return research_overview
    
    async def start_interactive_mode(self) -> None:
        """Start the system in interactive mode."""
        print("Welcome to Watson Co-Scientist interactive mode!")
        print("Enter a research goal to begin, or type 'exit' to quit.")
        
        while True:
            user_input = input("> ")
            
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting interactive mode.")
                break
                
            if user_input.lower() == "help":
                self._print_help()
                continue
                
            if user_input.startswith("goal:"):
                # Set a new research goal
                research_goal_text = user_input[5:].strip()
                print(f"Analyzing research goal: {research_goal_text[:50]}...")
                
                await self.analyze_research_goal(research_goal_text)
                print("Research goal set. Type 'run' to start processing, or 'run N' to run N iterations.")
                
            elif user_input.startswith("run"):
                # Run iterations
                if not self.current_research_goal:
                    print("No research goal set. Please set a goal first.")
                    continue
                    
                # Parse number of iterations
                parts = user_input.split()
                iterations = 1 if len(parts) == 1 else int(parts[1])
                
                print(f"Running {iterations} iteration(s)...")
                for i in range(iterations):
                    await self.run_iteration()
                    print(f"Completed iteration {i+1}/{iterations}")
                    
                # Print current state
                self._print_state()
                
            elif user_input == "state":
                # Print current state
                self._print_state()
                
            elif user_input == "overview":
                # Generate and print research overview
                if not self.current_research_goal:
                    print("No research goal set. Please set a goal first.")
                    continue
                    
                print("Generating research overview...")
                overview = await self._generate_research_overview()
                
                print("\n============ RESEARCH OVERVIEW ============")
                print(f"Title: {overview.title}")
                print(f"\nSummary: {overview.summary}")
                print("\nResearch Areas:")
                for i, area in enumerate(overview.research_areas, 1):
                    print(f"  {i}. {area.get('name', '')}")
                    print(f"     {area.get('description', '')[:100]}...")
                
            elif user_input == "protocols":
                # List experimental protocols
                if not self.current_research_goal:
                    print("No research goal set. Please set a goal first.")
                    continue
                
                # Get all protocols
                protocols = self.db.experimental_protocols.get_all()
                
                if not protocols:
                    print("No experimental protocols have been generated yet.")
                    continue
                
                print("\n============ EXPERIMENTAL PROTOCOLS ============")
                for i, protocol in enumerate(protocols, 1):
                    # Get the hypothesis for this protocol
                    hypothesis = self.db.hypotheses.get(protocol.hypothesis_id)
                    hypothesis_title = hypothesis.title if hypothesis else "Unknown hypothesis"
                    
                    # Get protocol reviews
                    reviews = self.db.get_reviews_for_protocol(protocol.id)
                    avg_score = sum(r.overall_score for r in reviews) / len(reviews) if reviews else "No reviews"
                    
                    print(f"{i}. {protocol.title}")
                    print(f"   For hypothesis: {hypothesis_title}")
                    print(f"   Average review score: {avg_score}")
                    print(f"   Steps: {len(protocol.steps)}")
                    print(f"   Materials: {len(protocol.materials)}")
                    print(f"   Created: {protocol.created_at.strftime('%Y-%m-%d %H:%M')}")
                    print()
                
            elif user_input == "generate-protocol":
                # Generate a new protocol
                if not self.current_research_goal:
                    print("No research goal set. Please set a goal first.")
                    continue
                
                print("Generating experimental protocol for a top hypothesis...")
                await self._generate_protocol()
                print("Protocol generation completed.")
                
            else:
                print("Unknown command. Type 'help' for a list of commands.")
    
    def _print_help(self) -> None:
        """Print help information."""
        print("\nCommands:")
        print("  goal: <text>     - Set a new research goal")
        print("  run [N]          - Run 1 or N iterations")
        print("  state            - Print the current state")
        print("  overview         - Generate and print a research overview")
        print("  protocols        - Show generated experimental protocols")
        print("  generate-protocol- Generate an experimental protocol for a top hypothesis")
        print("  help             - Print this help message")
        print("  exit             - Exit interactive mode")
    
    def _print_state(self) -> None:
        """Print the current state."""
        if not self.current_research_goal:
            print("No research goal set.")
            return
            
        print("\n============ CURRENT STATE ============")
        print(f"Research Goal: {self.current_research_goal.text[:100]}...")
        print(f"Iterations completed: {self.current_state['iterations_completed']}")
        print(f"Hypotheses generated: {self.current_state['num_hypotheses']}")
        print(f"Reviews completed: {self.current_state['num_reviews']}")
        print(f"Tournament matches: {self.current_state['num_tournament_matches']}")
        print(f"Experimental protocols: {self.current_state['num_protocols']}")
        
        # Print top hypotheses
        if self.current_state['top_hypotheses']:
            print("\nTop Hypotheses:")
            for i, h_id in enumerate(self.current_state['top_hypotheses'][:5], 1):
                hypothesis = self.db.hypotheses.get(h_id)
                if hypothesis:
                    print(f"  {i}. {hypothesis.title} (Rating: {hypothesis.elo_rating:.1f})")
        
        # Print agent weights for next iteration
        print("\nAgent weights for next iteration:")
        for agent, weight in self.agent_weights.items():
            print(f"  {agent}: {weight:.2f}")
    
    async def _generate_hypothesis(self) -> None:
        """Generate a new hypothesis."""
        try:
            # Choose generation method based on state
            if self.current_state["iterations_completed"] == 0:
                # First iteration: use basic generation
                hypotheses = await self.generation.generate_initial_hypotheses(
                    self.current_research_goal,
                    num_hypotheses=2
                )
            elif random.random() < 0.3:
                # Sometimes use debate
                hypotheses = await self.generation.generate_hypotheses_debate(
                    self.current_research_goal
                )
            else:
                # Use literature-based generation
                hypotheses = await self.generation.generate_hypotheses_with_literature(
                    self.current_research_goal,
                    num_hypotheses=2
                )
            
            # Save hypotheses to database
            for hypothesis in hypotheses:
                self.db.hypotheses.save(hypothesis)
                
            logger.info(f"Generated {len(hypotheses)} new hypotheses")
            
        except Exception as e:
            logger.error(f"Error generating hypothesis: {e}", exc_info=True)
    
    async def _review_hypothesis(self, hypothesis: Hypothesis) -> None:
        """
        Review a hypothesis using a mix of review types including enhanced deep verification
        and simulation approaches.
        
        Args:
            hypothesis (Hypothesis): The hypothesis to review.
        """
        try:
            # Get existing reviews for this hypothesis to determine what types we need
            existing_reviews = self.db.get_reviews_for_hypothesis(hypothesis.id)
            existing_review_types = set(review.review_type for review in existing_reviews)
            
            # Determine the iteration and system state
            iteration = self.current_state["iterations_completed"]
            top_hypothesis = hypothesis.id in self.current_state.get("top_hypotheses", [])
            elo_rating = hypothesis.elo_rating if hypothesis.elo_rating is not None else 1200
            
            # Choose review type based on state and what's missing
            
            # For new hypotheses, always start with a full review
            if "full" not in existing_review_types:
                review = await self.reflection.full_review(
                    hypothesis,
                    self.current_research_goal
                )
            
            # For hypotheses with better ratings that don't have a deep verification review, add one
            elif "deep_verification" not in existing_review_types and (top_hypothesis or elo_rating > 1250):
                review = await self.reflection.deep_verification_review(
                    hypothesis,
                    self.current_research_goal
                )
                
            # For top hypotheses without simulation reviews, do a simulation review
            elif "simulation" not in existing_review_types and (top_hypothesis or elo_rating > 1280):
                review = await self.reflection.simulation_review(
                    hypothesis,
                    self.current_research_goal
                )
            
            # For hypotheses that could benefit from observation reviews
            elif "observation" not in existing_review_types and iteration > 1:
                review = await self.reflection.observation_review(
                    hypothesis,
                    self.current_research_goal
                )
            
            # For hypotheses that already have basic reviews, choose randomly with higher weight
            # to deep verification and simulation for later iterations
            else:
                r = random.random()
                
                if iteration < 2:
                    # Early iterations focus on basic reviews
                    if r < 0.5:
                        review = await self.reflection.full_review(
                            hypothesis,
                            self.current_research_goal
                        )
                    elif r < 0.8:
                        review = await self.reflection.deep_verification_review(
                            hypothesis,
                            self.current_research_goal
                        )
                    else:
                        review = await self.reflection.observation_review(
                            hypothesis,
                            self.current_research_goal
                        )
                else:
                    # Later iterations emphasize deep verification and simulation
                    if r < 0.3:
                        review = await self.reflection.full_review(
                            hypothesis,
                            self.current_research_goal
                        )
                    elif r < 0.6:
                        review = await self.reflection.deep_verification_review(
                            hypothesis,
                            self.current_research_goal
                        )
                    elif r < 0.85:
                        review = await self.reflection.simulation_review(
                            hypothesis,
                            self.current_research_goal
                        )
                    else:
                        review = await self.reflection.observation_review(
                            hypothesis,
                            self.current_research_goal
                        )
            
            # Save review to database
            self.db.reviews.save(review)
            
            # Update hypothesis status
            hypothesis.status = HypothesisStatus.REVIEWED
            
            # Update hypothesis scores based on review
            if review.novelty_score is not None:
                hypothesis.novelty_score = review.novelty_score
            if review.correctness_score is not None:
                hypothesis.correctness_score = review.correctness_score
            if review.testability_score is not None:
                hypothesis.testability_score = review.testability_score
                
            # Update hypothesis metadata with verification and simulation insights
            if review.review_type == "deep_verification" and review.metadata:
                # Add probability assessment to hypothesis metadata if available
                prob_correct = review.metadata.get("probability_correct")
                if prob_correct is not None:
                    if not hypothesis.metadata:
                        hypothesis.metadata = {}
                    hypothesis.metadata["probability_correct"] = prob_correct
                    
                # Add verification experiments to metadata
                verification_exps = review.metadata.get("verification_experiments", [])
                if verification_exps:
                    if not hypothesis.metadata:
                        hypothesis.metadata = {}
                    hypothesis.metadata["verification_experiments"] = verification_exps
                    
            # Update with simulation insights
            if review.review_type == "simulation" and review.metadata:
                # Add predictions to hypothesis metadata
                predictions = review.metadata.get("predictions", [])
                if predictions:
                    if not hypothesis.metadata:
                        hypothesis.metadata = {}
                    hypothesis.metadata["predictions"] = predictions
                    
                # Add emergent properties to metadata
                emergent_props = review.metadata.get("emergent_properties", [])
                if emergent_props:
                    if not hypothesis.metadata:
                        hypothesis.metadata = {}
                    hypothesis.metadata["emergent_properties"] = emergent_props
                
            # Save updated hypothesis
            self.db.hypotheses.save(hypothesis)
            
            logger.info(f"Completed {review.review_type} review for hypothesis {hypothesis.id}")
            
        except Exception as e:
            logger.error(f"Error reviewing hypothesis {hypothesis.id}: {e}", exc_info=True)
    
    async def _run_tournament(self) -> None:
        """Run a tournament to compare and rank hypotheses."""
        try:
            # Get all hypotheses with at least one review
            all_hypotheses = self.db.hypotheses.get_all()
            reviewed_hypotheses = [h for h in all_hypotheses if h.status == HypothesisStatus.REVIEWED]
            
            if len(reviewed_hypotheses) < 2:
                logger.info("Not enough reviewed hypotheses for a tournament")
                return
                
            # Select pairs for tournament
            pairs = self.ranking.select_pairs_for_tournament(
                reviewed_hypotheses,
                self.matches_played,
                self.similarity_groups,
                num_pairs=3
            )
            
            if not pairs:
                logger.info("No pairs selected for tournament")
                return
                
            # Run tournament matches
            for hypothesis1, hypothesis2 in pairs:
                # Determine if this should be a detailed match
                detailed = random.random() < 0.3  # 30% chance of a detailed match
                
                # Conduct match
                match = await self.ranking.conduct_match(
                    hypothesis1,
                    hypothesis2,
                    self.current_research_goal,
                    detailed=detailed
                )
                
                # Save match to database
                self.db.tournament_matches.save(match)
                
                # Update Elo ratings
                self.ranking.update_elo_ratings(match, hypothesis1, hypothesis2)
                
                # Save updated hypotheses
                self.db.hypotheses.save(hypothesis1)
                self.db.hypotheses.save(hypothesis2)
                
            logger.info(f"Completed tournament with {len(pairs)} matches")
            
        except Exception as e:
            logger.error(f"Error running tournament: {e}", exc_info=True)
    
    async def _update_similarity_groups(self) -> None:
        """Update similarity groups for hypotheses."""
        try:
            # Get all hypotheses
            all_hypotheses = self.db.hypotheses.get_all()
            
            if len(all_hypotheses) < 2:
                logger.info("Not enough hypotheses for similarity calculation")
                return
                
            # Limit to a reasonable number of comparisons
            max_comparisons = 10
            
            # Calculate similarity groups
            self.similarity_groups = await self.proximity.build_similarity_groups(
                all_hypotheses,
                self.current_research_goal,
                similarity_threshold=0.7,
                max_comparisons=max_comparisons
            )
            
            # Optional: also cluster the hypotheses
            if len(all_hypotheses) >= 5 and random.random() < 0.5:
                clusters = await self.proximity.cluster_hypotheses(
                    all_hypotheses,
                    self.current_research_goal,
                    num_clusters=min(5, len(all_hypotheses) // 2)
                )
                
                # Save cluster information to metadata
                for cluster_id, hypothesis_ids in clusters.items():
                    for h_id in hypothesis_ids:
                        hypothesis = self.db.hypotheses.get(h_id)
                        if hypothesis:
                            hypothesis.metadata["cluster_id"] = cluster_id
                            self.db.hypotheses.save(hypothesis)
                
                logger.info(f"Clustered hypotheses into {len(clusters)} clusters")
                
            logger.info(f"Updated similarity groups for {len(all_hypotheses)} hypotheses")
            
        except Exception as e:
            logger.error(f"Error updating similarity groups: {e}", exc_info=True)
    
    async def _evolve_hypothesis(self) -> None:
        """Evolve a hypothesis to improve it."""
        try:
            # Get all hypotheses
            all_hypotheses = self.db.hypotheses.get_all()
            
            if not all_hypotheses:
                logger.info("No hypotheses available for evolution")
                return
                
            # Choose evolution method based on randomness
            r = random.random()
            
            if r < 0.4:
                # Improve a single hypothesis
                # Select a hypothesis with preference for higher ratings
                sorted_hypotheses = sorted(all_hypotheses, key=lambda h: h.elo_rating, reverse=True)
                target_idx = int(random.triangular(0, len(sorted_hypotheses) - 1, 0))
                hypothesis = sorted_hypotheses[target_idx]
                
                # Get reviews for this hypothesis
                reviews = self.db.get_reviews_for_hypothesis(hypothesis.id)
                
                # Use enhanced evolution strategy that automatically selects the appropriate
                # improvement strategy based on the hypothesis and reviews
                improved = await self.evolution.evolve_hypothesis(
                    hypothesis,
                    self.current_research_goal,
                    reviews
                )
                
                # Save the improved hypothesis
                self.db.hypotheses.save(improved)
                
                logger.info(f"Improved hypothesis {hypothesis.id} -> {improved.id}")
                
            elif r < 0.7:
                # Combine multiple hypotheses
                # Select 2-3 hypotheses, with preference for higher ratings
                sorted_hypotheses = sorted(all_hypotheses, key=lambda h: h.elo_rating, reverse=True)
                num_to_combine = min(2 + int(random.random() * 2), len(sorted_hypotheses))
                
                # Select hypotheses with preference for higher ratings
                indices = [int(random.triangular(0, len(sorted_hypotheses) - 1, 0)) for _ in range(num_to_combine)]
                candidates = [sorted_hypotheses[i] for i in indices]
                
                # Combine the hypotheses
                combined = await self.evolution.combine_hypotheses(
                    candidates,
                    self.current_research_goal
                )
                
                # Save the combined hypothesis
                self.db.hypotheses.save(combined)
                
                logger.info(f"Combined {num_to_combine} hypotheses -> {combined.id}")
                
            elif r < 0.85:
                # Generate an out-of-box hypothesis
                out_of_box = await self.evolution.generate_out_of_box_hypothesis(
                    self.current_research_goal,
                    existing_hypotheses=all_hypotheses
                )
                
                # Save the out-of-box hypothesis
                self.db.hypotheses.save(out_of_box)
                
                logger.info(f"Generated out-of-box hypothesis: {out_of_box.id}")
                
            else:
                # Simplify a complex hypothesis
                # Find a long, complex hypothesis
                complex_candidates = [h for h in all_hypotheses if len(h.description) > 500]
                
                if complex_candidates:
                    complex_hypothesis = random.choice(complex_candidates)
                    
                    # Simplify the hypothesis
                    simplified = await self.evolution.simplify_hypothesis(
                        complex_hypothesis,
                        self.current_research_goal
                    )
                    
                    # Save the simplified hypothesis
                    self.db.hypotheses.save(simplified)
                    
                    logger.info(f"Simplified hypothesis {complex_hypothesis.id} -> {simplified.id}")
                else:
                    logger.info("No complex hypotheses found for simplification")
            
        except Exception as e:
            logger.error(f"Error evolving hypothesis: {e}", exc_info=True)
    
    async def _generate_meta_review(self) -> MetaReview:
        """
        Generate a meta-review synthesizing insights from all reviews.
        
        Returns:
            MetaReview: The generated meta-review.
        """
        try:
            # Get all reviews
            all_reviews = self.db.reviews.get_all()
            
            if not all_reviews:
                logger.info("No reviews available for meta-review")
                return None
                
            # Generate meta-review
            meta_review = await self.meta_review.synthesize_reviews(
                all_reviews,
                self.current_research_goal
            )
            
            # Save meta-review to database
            self.db.meta_reviews.save(meta_review)
            
            # Also analyze tournament results if we have enough matches
            all_matches = self.db.tournament_matches.get_all()
            if len(all_matches) >= 5:
                # Create a dictionary of hypotheses by ID
                all_hypotheses = self.db.hypotheses.get_all()
                hypotheses_dict = {h.id: h for h in all_hypotheses}
                
                # Analyze tournament results
                tournament_analysis = await self.meta_review.analyze_tournament_results(
                    all_matches,
                    hypotheses_dict,
                    self.current_research_goal
                )
                
                # Store analysis in meta-review metadata
                meta_review.metadata = {"tournament_analysis": tournament_analysis}
                self.db.meta_reviews.save(meta_review)
                
            logger.info(f"Generated meta-review for research goal {self.current_research_goal.id}")
            return meta_review
            
        except Exception as e:
            logger.error(f"Error generating meta-review: {e}", exc_info=True)
            return None
    
    async def _generate_research_overview(self) -> ResearchOverview:
        """
        Generate a comprehensive research overview incorporating deep verification and
        simulation insights for enhanced scientific rigor.
        
        Returns:
            ResearchOverview: The generated research overview.
        """
        try:
            # Get top hypotheses
            all_hypotheses = self.db.hypotheses.get_all()
            sorted_hypotheses = sorted(all_hypotheses, key=lambda h: h.elo_rating, reverse=True)
            top_hypotheses = sorted_hypotheses[:10]
            
            if not top_hypotheses:
                logger.info("No hypotheses available for research overview")
                return None
                
            # Get latest meta-review
            meta_review = self.db.get_latest_meta_review(self.current_research_goal.id)
            
            # Get tournament analysis
            tournament_analysis = None
            if meta_review and meta_review.metadata and "tournament_analysis" in meta_review.metadata:
                tournament_analysis = meta_review.metadata["tournament_analysis"]
                
            # Get protocols for analysis
            all_protocols = self.db.experimental_protocols.get_all()
            
            # Create dictionary of hypotheses by ID
            all_hypotheses = self.db.hypotheses.get_all()
            hypotheses_dict = {h.id: h for h in all_hypotheses}
            
            # Get protocol analysis if we have protocols
            protocol_analysis = None
            if all_protocols:
                protocol_analysis = await self.meta_review.analyze_protocols(
                    all_protocols,
                    hypotheses_dict,
                    self.current_research_goal
                )
            
            # Get verification and simulation reviews for top hypotheses
            verification_analysis = None
            
            # Get all deep verification and simulation reviews
            all_reviews = self.db.reviews.get_all()
            deep_verification_reviews = [r for r in all_reviews if r.review_type == "deep_verification"]
            simulation_reviews = [r for r in all_reviews if r.review_type == "simulation"]
            
            # Only proceed with verification analysis if we have enough reviews
            if len(deep_verification_reviews) + len(simulation_reviews) >= 3:
                verification_analysis = await self.meta_review.analyze_verification_reviews(
                    deep_verification_reviews,
                    simulation_reviews,
                    self.current_research_goal
                )
                
                # Add logging about what was found
                logger.info(f"Analyzed {len(deep_verification_reviews)} deep verification reviews and "
                           f"{len(simulation_reviews)} simulation reviews for verification analysis")
                
                if verification_analysis:
                    num_patterns = len(verification_analysis.get("causal_reasoning_patterns", []))
                    num_experiments = len(verification_analysis.get("verification_experiments", []))
                    logger.info(f"Found {num_patterns} causal reasoning patterns and {num_experiments} verification experiments")
                
            # Generate enhanced research overview with verification insights
            overview = await self.meta_review.generate_research_overview(
                self.current_research_goal,
                top_hypotheses,
                meta_review,
                tournament_analysis,
                protocol_analysis,
                verification_analysis
            )
            
            # Save research overview to database
            self.db.research_overviews.save(overview)
            
            # Update state
            self.current_state["last_research_overview_id"] = overview.id
            
            logger.info(f"Generated research overview for research goal {self.current_research_goal.id}")
            return overview
            
        except Exception as e:
            logger.error(f"Error generating research overview: {e}", exc_info=True)
            return None
    
    async def _generate_protocol(self) -> None:
        """Generate an experimental protocol for a high-ranking hypothesis."""
        try:
            # Get hypotheses that need protocols
            hypotheses_needing_protocols = self.db.get_hypotheses_needing_protocols(
                self.current_research_goal.id, 
                limit=5
            )
            
            if not hypotheses_needing_protocols:
                logger.info("No hypotheses need experimental protocols at this time")
                
                # If no hypotheses need protocols, maybe generate a hypothesis with a protocol
                if random.random() < 0.3:
                    hypothesis, protocol = await self.generation.generate_hypothesis_with_protocol(
                        self.current_research_goal
                    )
                    
                    if hypothesis and protocol:
                        self.db.hypotheses.save(hypothesis)
                        self.db.experimental_protocols.save(protocol)
                        logger.info(f"Generated new hypothesis {hypothesis.id} with integrated protocol {protocol.id}")
                    
                return
                
            # Choose a hypothesis, weighted by rating
            ratings = [h.elo_rating for h in hypotheses_needing_protocols]
            total_rating = sum(ratings)
            if total_rating > 0:
                probs = [r / total_rating for r in ratings]
                hypothesis = random.choices(hypotheses_needing_protocols, weights=probs, k=1)[0]
            else:
                hypothesis = random.choice(hypotheses_needing_protocols)
            
            # Generate protocol
            protocol = await self.generation.generate_experimental_protocol(
                hypothesis, 
                self.current_research_goal
            )
            
            if protocol:
                # Save protocol to database
                self.db.experimental_protocols.save(protocol)
                logger.info(f"Generated experimental protocol {protocol.id} for hypothesis {hypothesis.id}")
            else:
                logger.warning(f"Failed to generate protocol for hypothesis {hypothesis.id}")
                
        except Exception as e:
            logger.error(f"Error generating experimental protocol: {e}", exc_info=True)
            
    async def _review_protocol(self, protocol: ExperimentalProtocol, hypothesis: Hypothesis) -> None:
        """
        Review an experimental protocol.
        
        Args:
            protocol (ExperimentalProtocol): The protocol to review.
            hypothesis (Hypothesis): The hypothesis the protocol is testing.
        """
        try:
            # Generate the protocol review
            review = await self.reflection.review_protocol(
                protocol,
                hypothesis,
                self.current_research_goal
            )
            
            # Save the review to the database
            self.db.reviews.save(review)
            
            logger.info(f"Reviewed protocol {protocol.id} for hypothesis {hypothesis.id}")
            
        except Exception as e:
            logger.error(f"Error reviewing protocol {protocol.id}: {e}", exc_info=True)
            
    def _get_unreviewed_protocols(self) -> List[Dict[str, Any]]:
        """
        Get protocols that have not been reviewed.
        
        Returns:
            List[Dict[str, Any]]: List of dictionaries with protocol and corresponding hypothesis.
        """
        try:
            all_protocols = self.db.experimental_protocols.get_all()
            result = []
            
            for protocol in all_protocols:
                # Check if this protocol has any reviews
                protocol_reviews = self.db.get_reviews_for_protocol(protocol.id)
                
                if not protocol_reviews:
                    # This protocol hasn't been reviewed - get the hypothesis
                    hypothesis = self.db.hypotheses.get(protocol.hypothesis_id)
                    
                    if hypothesis:
                        result.append({
                            "protocol": protocol,
                            "hypothesis": hypothesis
                        })
            
            # Sort by creation time (newest first)
            result.sort(key=lambda x: x["protocol"].created_at, reverse=True)
            return result
            
        except Exception as e:
            logger.error(f"Error getting unreviewed protocols: {e}")
            return []
    
    def _get_unreviewed_hypotheses(self) -> List[Hypothesis]:
        """
        Get hypotheses that have not been reviewed.
        Prioritizes user-submitted hypotheses first, followed by system-generated ones.
        
        Returns:
            List[Hypothesis]: Unreviewed hypotheses, prioritized by source and creation time.
        """
        all_hypotheses = self.db.hypotheses.get_all()
        unreviewed = [h for h in all_hypotheses if h.status == HypothesisStatus.GENERATED]
        
        # Separate user-submitted and system-generated hypotheses
        user_hypotheses = [h for h in unreviewed if h.source == HypothesisSource.USER]
        system_hypotheses = [h for h in unreviewed if h.source != HypothesisSource.USER]
        
        # Sort user hypotheses by creation time (newest first)
        user_hypotheses.sort(key=lambda h: h.created_at, reverse=True)
        
        # Sort system hypotheses by creation time
        system_hypotheses.sort(key=lambda h: h.created_at)
        
        # Return prioritized list: user hypotheses first, then system hypotheses
        return user_hypotheses + system_hypotheses