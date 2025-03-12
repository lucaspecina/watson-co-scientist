"""
Core system implementation for the Co-Scientist system.
Coordinates agents and manages the overall workflow.
"""

import os
import json
import uuid
import logging
import asyncio
import random
from typing import Dict, List, Any, Optional, Set, Tuple, Type
from datetime import datetime
from enum import Enum, auto

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
    HypothesisSource,
    UserFeedback,
    ResearchFocus,
    UserSession
)
from ..config.config import load_config, SystemConfig
from ..agents.supervisor_agent import SupervisorAgent
from ..agents.generation_agent import GenerationAgent
from ..agents.reflection_agent import ReflectionAgent
from ..agents.ranking_agent import RankingAgent
from ..agents.proximity_agent import ProximityAgent
from ..agents.evolution_agent import EvolutionAgent
from ..agents.meta_review_agent import MetaReviewAgent
# Import domain knowledge components
from ..tools.domain_specific.knowledge_manager import DomainKnowledgeManager
from ..tools.domain_specific.cross_domain_synthesizer import CrossDomainSynthesizer
from ..utils.logger import user_output

logger = logging.getLogger("co_scientist")

class SessionState(str, Enum):
    """Enum representing the different states of a research session."""
    INITIALIZING = "initializing"
    ANALYZING_GOAL = "analyzing_goal"
    GENERATING_HYPOTHESES = "generating_hypotheses"
    WAITING_FOR_FEEDBACK = "waiting_for_feedback"
    PROCESSING_FEEDBACK = "processing_feedback"
    EVOLVING_HYPOTHESES = "evolving_hypotheses"
    GENERATING_OVERVIEW = "generating_overview"
    COMPLETED = "completed"
    PAUSED = "paused"
    ERROR = "error"

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
        
        # Initialize domain knowledge manager
        self.domain_knowledge = DomainKnowledgeManager(self.config.get("domain_knowledge", {}))
        
        # Initialize agents
        self.supervisor = SupervisorAgent(self.config)
        self.generation = GenerationAgent(self.config)
        self.reflection = ReflectionAgent(self.config)
        self.ranking = RankingAgent(self.config)
        self.proximity = ProximityAgent(self.config)
        self.evolution = EvolutionAgent(self.config)
        self.meta_review = MetaReviewAgent(self.config)
        
        # Initialize cross-domain synthesizer
        self.cross_domain_synthesizer = CrossDomainSynthesizer(self.domain_knowledge)
        
        # Initialize state
        self.current_research_goal: Optional[ResearchGoal] = None
        self.research_plan_config: Dict[str, Any] = {}
        self.agent_weights: Dict[str, float] = {}
        self.current_state: Dict[str, Any] = {}
        self.similarity_groups: Dict[str, Set[str]] = {}
        self.matches_played: Dict[Tuple[str, str], bool] = {}
        self.session_state: SessionState = SessionState.INITIALIZING
        self.current_user_session: Optional[UserSession] = None
        self.current_user_id: str = str(uuid.uuid4())
        
        logger.info(f"Initialized Co-Scientist system with configuration '{config_name}'")
    
    async def analyze_research_goal(self, research_goal_text: str, input_files: List[str] = None, urls: List[str] = None) -> ResearchGoal:
        """
        Analyze a research goal and initialize the system for processing it.
        
        Args:
            research_goal_text (str): The research goal text.
            input_files (List[str], optional): List of local file paths to process.
            urls (List[str], optional): List of URLs to retrieve and process.
            
        Returns:
            ResearchGoal: The created research goal.
        """
        logger.info(f"Analyzing research goal: {research_goal_text[:100]}...")
        self.session_state = SessionState.ANALYZING_GOAL
        
        # Create the research goal
        self.current_research_goal = ResearchGoal(
            text=research_goal_text,
            preferences={},
            constraints={}
        )
        
        # Save to database
        self.db.research_goals.save(self.current_research_goal)
        
        # Create a new user session
        self.current_user_session = UserSession(
            user_id=self.current_user_id,
            research_goal_id=self.current_research_goal.id,
            active=True,
            metadata={"creation_method": "analyze_research_goal"}
        )
        self.db.user_sessions.save(self.current_user_session)
        
        # Process input files if provided
        if input_files:
            await self._process_input_files(input_files)
            
        # Process URLs if provided
        if urls:
            await self._process_input_urls(urls)
        
        # Parse the research goal
        self.research_plan_config = await self.supervisor.parse_research_goal(research_goal_text)
        
        # Update the research goal with preferences and constraints
        self.current_research_goal.preferences = self.research_plan_config.get("preferences", {})
        self.current_research_goal.constraints = {"constraints": self.research_plan_config.get("constraints", [])}
        
        # Save the updated research goal
        self.db.research_goals.save(self.current_research_goal)
        
        # Initialize domain knowledge for this research goal
        relevant_domains = await self._initialize_domain_knowledge(research_goal_text)
        
        # Initialize state
        self.current_state = {
            "num_hypotheses": 0,
            "num_reviews": 0,
            "num_tournament_matches": 0,
            "num_protocols": 0,
            "top_hypotheses": [],
            "iterations_completed": 0,
            "last_research_overview_id": None,
            "relevant_domains": relevant_domains,
            "feedback_incorporated": False,
            "focus_areas_count": 0,
            "resources_processed": 0,
            "processed_files": [],
            "processed_urls": []
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
        
        # Update session state
        self.session_state = SessionState.GENERATING_HYPOTHESES
        
        return self.current_research_goal
    
    async def _process_input_files(self, file_paths: List[str]) -> None:
        """
        Process input files provided by the user.
        
        Args:
            file_paths (List[str]): List of local file paths to process.
        """
        logger.info(f"Processing {len(file_paths)} input files")
        
        processed_files = []
        for file_path in file_paths:
            try:
                # Only process the file if it exists
                if not os.path.exists(file_path):
                    logger.warning(f"File not found: {file_path}")
                    continue
                
                file_ext = os.path.splitext(file_path)[1].lower()
                
                # Process the file based on its extension
                if file_ext in ['.pdf', '.PDF']:
                    # Process PDF file
                    logger.info(f"Processing PDF file: {file_path}")
                    # Add extraction logic or call to appropriate tool
                    extracted_info = await self._extract_knowledge_from_pdf(file_path)
                    processed_files.append({"path": file_path, "type": "pdf", "status": "processed"})
                    
                elif file_ext in ['.txt', '.md', '.rst']:
                    # Process text file
                    logger.info(f"Processing text file: {file_path}")
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    # Process the text content
                    await self._process_text_content(content, source=f"file:{file_path}")
                    processed_files.append({"path": file_path, "type": "text", "status": "processed"})
                    
                elif file_ext in ['.csv', '.xlsx', '.xls']:
                    # Process data file
                    logger.info(f"Processing data file: {file_path}")
                    # Add data processing logic
                    processed_files.append({"path": file_path, "type": "data", "status": "processed"})
                    
                else:
                    logger.warning(f"Unsupported file type: {file_ext} for file {file_path}")
                    processed_files.append({"path": file_path, "type": "unknown", "status": "skipped"})
                    
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
                processed_files.append({"path": file_path, "status": "error", "error": str(e)})
        
        # Update the current state with processed files
        self.current_state["processed_files"] = processed_files
        self.current_state["resources_processed"] += len([f for f in processed_files if f["status"] == "processed"])
        
    async def _process_input_urls(self, urls: List[str]) -> None:
        """
        Process URLs provided by the user.
        
        Args:
            urls (List[str]): List of URLs to retrieve and process.
        """
        logger.info(f"Processing {len(urls)} input URLs")
        
        processed_urls = []
        for url in urls:
            try:
                logger.info(f"Processing URL: {url}")
                
                # Add URL processing logic here based on the URL type
                if url.endswith(('.pdf', '.PDF')):
                    # Process PDF URL
                    extracted_info = await self._extract_knowledge_from_pdf_url(url)
                    processed_urls.append({"url": url, "type": "pdf", "status": "processed"})
                    
                elif url.endswith(('.html', '.htm', '')):
                    # Process web page
                    extracted_info = await self._extract_knowledge_from_webpage(url)
                    processed_urls.append({"url": url, "type": "webpage", "status": "processed"})
                    
                else:
                    # Generic URL processing
                    extracted_info = await self._extract_knowledge_from_generic_url(url)
                    processed_urls.append({"url": url, "type": "generic", "status": "processed"})
                    
            except Exception as e:
                logger.error(f"Error processing URL {url}: {e}")
                processed_urls.append({"url": url, "status": "error", "error": str(e)})
        
        # Update the current state with processed URLs
        self.current_state["processed_urls"] = processed_urls
        self.current_state["resources_processed"] += len([u for u in processed_urls if u["status"] == "processed"])
        
    async def _extract_knowledge_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Extract knowledge from a local PDF file."""
        # Placeholder - this would call PDF processing tools
        return {"source": pdf_path, "status": "processed"}
    
    async def _extract_knowledge_from_pdf_url(self, pdf_url: str) -> Dict[str, Any]:
        """Extract knowledge from a PDF URL."""
        # Placeholder - this would download and process the PDF
        return {"source": pdf_url, "status": "processed"}
    
    async def _extract_knowledge_from_webpage(self, url: str) -> Dict[str, Any]:
        """Extract knowledge from a webpage."""
        # Placeholder - this would scrape and process the webpage
        return {"source": url, "status": "processed"}
    
    async def _extract_knowledge_from_generic_url(self, url: str) -> Dict[str, Any]:
        """Extract knowledge from a generic URL."""
        # Placeholder - generic URL processing
        return {"source": url, "status": "processed"}
    
    async def _process_text_content(self, content: str, source: str) -> None:
        """Process text content from a file or other source."""
        # Placeholder - this would process the text content
        logger.info(f"Processing text content from {source} ({len(content)} characters)")
        
    async def _initialize_domain_knowledge(self, research_goal_text: str) -> List[str]:
        """
        Initialize domain knowledge for a research goal.
        
        Args:
            research_goal_text (str): The research goal text.
            
        Returns:
            List[str]: List of relevant domains.
        """
        # Detect relevant domains
        try:
            # Use cross_domain_synthesizer to detect relevant domains
            domain_info = self.cross_domain_synthesizer.detect_research_domains(research_goal_text)
            relevant_domains = [domain for domain, score in domain_info if score > 0.4]
            
            logger.info(f"Detected relevant domains: {', '.join(relevant_domains)}")
            
            # Initialize domain-specific knowledge for each relevant domain
            if relevant_domains:
                await self.domain_knowledge.initialize(domains=relevant_domains)
            else:
                # Fall back to default domains if none detected
                default_domains = ["biomedicine", "biology", "chemistry"]
                logger.info(f"Using default domains: {', '.join(default_domains)}")
                await self.domain_knowledge.initialize(domains=default_domains)
                relevant_domains = default_domains
                
            return relevant_domains
        except Exception as e:
            logger.error(f"Error initializing domain knowledge: {e}")
            return ["general"]  # Default to general domain if detection fails
    
    async def load_research_goal(self, goal_id: str) -> Optional[ResearchGoal]:
        """
        Load an existing research goal by ID.
        
        Args:
            goal_id (str): The ID of the research goal to load.
            
        Returns:
            Optional[ResearchGoal]: The loaded research goal, or None if not found.
        """
        logger.info(f"Loading research goal with ID: {goal_id}")
        
        self.session_state = SessionState.INITIALIZING
        
        # Get the research goal from the database
        research_goal = self.db.research_goals.get(goal_id)
        if not research_goal:
            logger.error(f"Research goal with ID {goal_id} not found.")
            return None
        
        self.current_research_goal = research_goal
        
        # Check for an existing user session for this goal
        existing_sessions = self.db.user_sessions.find_by("research_goal_id", goal_id)
        active_sessions = [s for s in existing_sessions if s.active]
        
        if active_sessions:
            # Use the most recent active session
            self.current_user_session = sorted(active_sessions, key=lambda s: s.session_start, reverse=True)[0]
            self.current_user_id = self.current_user_session.user_id
            logger.info(f"Resuming active session {self.current_user_session.id} for user {self.current_user_id}")
        else:
            # Create a new user session
            self.current_user_id = str(uuid.uuid4())
            self.current_user_session = UserSession(
                user_id=self.current_user_id,
                research_goal_id=goal_id,
                active=True,
                metadata={"creation_method": "load_research_goal"}
            )
            self.db.user_sessions.save(self.current_user_session)
            logger.info(f"Created new session {self.current_user_session.id} for user {self.current_user_id}")
        
        # Load existing hypotheses
        hypotheses = self.db.hypotheses.find_by("research_goal_id", goal_id)
        
        # Load existing reviews
        reviews = []
        for hypothesis in hypotheses:
            reviews.extend(self.db.get_reviews_for_hypothesis(hypothesis.id))
        
        # Load existing tournament matches
        tournament_matches = []
        for hypothesis in hypotheses:
            tournament_matches.extend(self.db.get_matches_for_hypothesis(hypothesis.id))
        
        # Load existing research overviews
        research_overview = self.db.get_latest_research_overview(goal_id)
        
        # Load existing focus areas
        focus_areas = self.db.get_active_research_focus(goal_id)
        
        # Rebuild the research plan config
        try:
            self.research_plan_config = await self.supervisor.parse_research_goal(research_goal.text)
        except Exception as e:
            logger.error(f"Error parsing research goal: {e}")
            self.research_plan_config = {"preferences": {}, "constraints": []}
        
        # Initialize state
        self.current_state = {
            "num_hypotheses": len(hypotheses),
            "num_reviews": len(reviews),
            "num_tournament_matches": len(tournament_matches),
            "num_protocols": len(self.db.get_protocols_for_goal(goal_id)),
            "top_hypotheses": self._get_top_hypotheses(hypotheses),
            "iterations_completed": max(0, self._get_iterations_completed(hypotheses, reviews, tournament_matches)),
            "last_research_overview_id": research_overview.id if research_overview else None,
            "relevant_domains": self._get_relevant_domains(hypotheses),
            "feedback_incorporated": self._has_incorporated_feedback(goal_id),
            "focus_areas_count": len(focus_areas),
            "resources_processed": self._count_processed_resources(goal_id),
            "processed_files": [],
            "processed_urls": []
        }
        
        # Initialize agent weights
        # Adjust weights based on the current state
        generation_weight = 0.3 if self.current_state["num_hypotheses"] > 10 else 0.65
        evolution_weight = 0.3 if self.current_state["num_hypotheses"] > 0 else 0.0
        
        self.agent_weights = {
            "generation": generation_weight,
            "reflection": 0.2,
            "ranking": 0.05,
            "proximity": 0.05,
            "evolution": evolution_weight,
            "meta_review": 0.05,
            "protocol_generation": 0.1
        }
        
        # Determine the current session state based on the loaded data
        if self.current_state["num_hypotheses"] > 0:
            if self.current_state["feedback_incorporated"]:
                self.session_state = SessionState.GENERATING_HYPOTHESES
            else:
                self.session_state = SessionState.WAITING_FOR_FEEDBACK
        else:
            self.session_state = SessionState.GENERATING_HYPOTHESES
            
        logger.info(f"Loaded research goal {goal_id} with {len(hypotheses)} hypotheses, {len(reviews)} reviews, and {len(tournament_matches)} tournament matches")
        return research_goal
    
    def _get_top_hypotheses(self, hypotheses: List[Hypothesis]) -> List[str]:
        """Get the IDs of the top hypotheses based on Elo rating."""
        sorted_hypotheses = sorted(hypotheses, key=lambda h: h.elo_rating, reverse=True)
        return [h.id for h in sorted_hypotheses[:5]]
    
    def _get_iterations_completed(self, hypotheses: List[Hypothesis], reviews: List[Review], tournament_matches: List[TournamentMatch]) -> int:
        """Estimate the number of iterations completed based on the data."""
        if not hypotheses:
            return 0
        
        # Simple heuristic: 1 iteration per 5 hypotheses, minimum 1 if any hypotheses exist
        return max(1, len(hypotheses) // 5)
    
    def _get_relevant_domains(self, hypotheses: List[Hypothesis]) -> List[str]:
        """Extract relevant domains from hypotheses metadata."""
        domains = set()
        for hypothesis in hypotheses:
            if "domains" in hypothesis.metadata:
                domains.update(hypothesis.metadata["domains"])
        
        # Default to general if no domains found
        return list(domains) if domains else ["general"]
    
    def _has_incorporated_feedback(self, research_goal_id: str) -> bool:
        """Check if any user feedback has been incorporated."""
        feedback = self.db.get_user_feedback(research_goal_id)
        return any(f.action_taken for f in feedback)
    
    def _count_processed_resources(self, research_goal_id: str) -> int:
        """Count the number of processed resources for this research goal."""
        # This could be more sophisticated by checking citations, etc.
        feedback = self.db.get_user_feedback(research_goal_id)
        return sum(len(f.resources) for f in feedback)
    
    async def run_iteration(self) -> Dict[str, Any]:
        """
        Run one iteration of the system.
        
        Returns:
            Dict[str, Any]: The updated system state.
        """
        if not self.current_research_goal:
            raise ValueError("No research goal set. Call analyze_research_goal first.")
            
        logger.info(f"Running iteration with session state: {self.session_state}")
        
        # Process any pending feedback if available and we're not already processing it
        if self.session_state != SessionState.PROCESSING_FEEDBACK:
            pending_feedback = self.db.get_user_feedback(
                self.current_research_goal.id, 
                require_action=True,
                action_taken=False
            )
            
            if pending_feedback and len(pending_feedback) > 0:
                logger.info(f"Found {len(pending_feedback)} pending feedback items to process")
                self.session_state = SessionState.PROCESSING_FEEDBACK
                await self._process_user_feedback(pending_feedback)
        
        # If we're waiting for feedback and no feedback is available, return current state
        if self.session_state == SessionState.WAITING_FOR_FEEDBACK:
            logger.info("System is waiting for user feedback. No actions taken.")
            return self.current_state
        
        # Decide what to do based on the current state
        if self.session_state == SessionState.PROCESSING_FEEDBACK:
            # After processing feedback, move to generating hypotheses
            self.session_state = SessionState.GENERATING_HYPOTHESES
        
        # Execute the main iteration tasks
        if self.session_state == SessionState.GENERATING_HYPOTHESES or self.session_state == SessionState.EVOLVING_HYPOTHESES:
            # Run the main iteration logic
            
            # Phase 1: Generate new hypotheses
            num_new_hypotheses = 0
            if random.random() < self.agent_weights["generation"]:
                logger.info("Generating new hypotheses")
                num_generated = await self._generate_hypothesis()
                num_new_hypotheses += num_generated
                logger.info(f"Generated {num_generated} new hypotheses")
            
            # Phase 2: Review hypotheses
            unreviewed = self._get_unreviewed_hypotheses()
            num_new_reviews = 0
            if unreviewed and random.random() < self.agent_weights["reflection"]:
                logger.info(f"Reviewing {len(unreviewed)} hypotheses")
                for hypothesis in unreviewed[:3]:  # Limit to 3 per iteration
                    await self._review_hypothesis(hypothesis)
                    num_new_reviews += 1
            
            # Phase 3: Run tournament
            if random.random() < self.agent_weights["ranking"]:
                logger.info("Running tournament")
                num_matches = await self._run_tournament()
                logger.info(f"Completed {num_matches} tournament matches")
            
            # Phase 4: Update similarity groups
            if random.random() < self.agent_weights["proximity"]:
                logger.info("Updating similarity groups")
                await self._update_similarity_groups()
            
            # Phase 5: Evolve hypotheses
            # Always try to evolve at least one hypothesis per iteration to ensure evolution happens regularly
            num_evolved = 0
            logger.info("Evolving hypotheses")
            num_evolved = await self._evolve_hypothesis()
            logger.info(f"Evolved {num_evolved} new hypotheses")
            num_new_hypotheses += num_evolved
            
            # Phase 6: Generate meta-review
            if random.random() < self.agent_weights["meta_review"]:
                logger.info("Generating meta-review")
                meta_review = await self._generate_meta_review()
                if meta_review is not None:
                    logger.info(f"Generated meta-review with {len(meta_review.common_issues)} common issues")
                else:
                    logger.info("No meta-review generated due to insufficient reviews")
            
            # Phase 7: Generate protocols
            if random.random() < self.agent_weights["protocol_generation"]:
                logger.info("Generating experimental protocols")
                await self._generate_protocol()
            
            # Update state
            all_hypotheses = self.db.get_hypotheses_for_goal(self.current_research_goal.id)
            all_reviews = []
            for hypothesis in all_hypotheses:
                all_reviews.extend(self.db.get_reviews_for_hypothesis(hypothesis.id))
            
            # Update the current state
            self.current_state["num_hypotheses"] = len(all_hypotheses)
            self.current_state["num_reviews"] = len(all_reviews)
            self.current_state["iterations_completed"] += 1
            
            # Get top hypotheses
            self.current_state["top_hypotheses"] = self._get_top_hypotheses(all_hypotheses)
            
            # After completing the generation phase, check if we should move to waiting for feedback
            if self.current_state["iterations_completed"] % 1 == 0:  # After each iteration
                # Generate research overview
                self.session_state = SessionState.GENERATING_OVERVIEW
                research_overview = await self._generate_research_overview()
                if research_overview:
                    logger.info(f"Generated research overview with {len(research_overview.research_areas)} research areas")
                    self.current_state["last_research_overview_id"] = research_overview.id
                
                # Now transition to waiting for feedback
                self.session_state = SessionState.WAITING_FOR_FEEDBACK
                logger.info("Iteration complete. System is now waiting for user feedback.")
            
        # Re-plan agent weights for next iteration
        self.agent_weights = await self.supervisor.plan_task_allocation(
            self.current_research_goal,
            self.research_plan_config,
            self.current_state
        )
        
        return self.current_state
    
    async def _process_user_feedback(self, feedback_items: List[UserFeedback]) -> None:
        """
        Process user feedback and incorporate it into the system.
        
        Args:
            feedback_items (List[UserFeedback]): The feedback items to process.
        """
        logger.info(f"Processing {len(feedback_items)} feedback items")
        
        for feedback in feedback_items:
            logger.info(f"Processing feedback: {feedback.id} of type {feedback.feedback_type}")
            
            try:
                # Process different types of feedback
                if feedback.feedback_type == "general":
                    # General feedback on the research direction
                    await self._incorporate_general_feedback(feedback)
                    
                elif feedback.feedback_type == "hypothesis":
                    # Feedback on a specific hypothesis
                    if feedback.hypothesis_id:
                        await self._incorporate_hypothesis_feedback(feedback)
                    
                elif feedback.feedback_type == "resource":
                    # Process resources provided by the user
                    await self._process_resource_feedback(feedback)
                    
                elif feedback.feedback_type == "focus":
                    # Process focus areas
                    await self._incorporate_focus_feedback(feedback)
                    
                elif feedback.feedback_type == "experimental":
                    # Process experimental results
                    await self._incorporate_experimental_results(feedback)
                
                # Mark the feedback as processed
                feedback.action_taken = True
                self.db.user_feedback.save(feedback)
                
                # Update the state to reflect feedback incorporation
                self.current_state["feedback_incorporated"] = True
                
            except Exception as e:
                logger.error(f"Error processing feedback {feedback.id}: {e}")
                # Mark as error in metadata but don't set action_taken
                feedback.metadata["processing_error"] = str(e)
                self.db.user_feedback.save(feedback)
    
    async def _incorporate_general_feedback(self, feedback: UserFeedback) -> None:
        """Incorporate general feedback into the research process."""
        # This would update the research plan config based on general feedback
        logger.info(f"Incorporating general feedback: {feedback.id}")
        
        # Update research plan configuration
        updates = await self.supervisor.update_research_plan(
            self.current_research_goal,
            self.research_plan_config,
            feedback.text
        )
        
        if updates:
            # Apply updates to the research plan config
            for key, value in updates.items():
                self.research_plan_config[key] = value
                
            logger.info(f"Updated research plan with {len(updates)} changes from feedback")
    
    async def _incorporate_hypothesis_feedback(self, feedback: UserFeedback) -> None:
        """Incorporate feedback on a specific hypothesis."""
        logger.info(f"Incorporating feedback for hypothesis: {feedback.hypothesis_id}")
        
        # Get the hypothesis
        hypothesis = self.db.hypotheses.get(feedback.hypothesis_id)
        if not hypothesis:
            logger.warning(f"Hypothesis {feedback.hypothesis_id} not found")
            return
            
        # Create a user review based on the feedback
        review = Review(
            hypothesis_id=hypothesis.id,
            review_type="user",
            reviewer=f"user-{feedback.user_id[:8]}",
            text=feedback.text,
            user_id=feedback.user_id,
            created_at=datetime.now(),
            critiques=[],
            strengths=[],
            improvement_suggestions=[]
        )
        
        # Add to database
        self.db.reviews.save(review)
        logger.info(f"Created review {review.id} from user feedback")
        
        # Queue the hypothesis for evolution in next iteration
        hypothesis.metadata["priority_for_evolution"] = True
        self.db.hypotheses.save(hypothesis)
    
    async def _process_resource_feedback(self, feedback: UserFeedback) -> None:
        """Process resource feedback (URLs, papers, files, etc.)"""
        logger.info(f"Processing resources from feedback: {feedback.id}")
        
        for resource in feedback.resources:
            resource_type = list(resource.keys())[0]
            resource_value = resource[resource_type]
            
            if resource_type == "url":
                # Process a URL
                logger.info(f"Processing URL from feedback: {resource_value}")
                try:
                    if resource_value.endswith(('.pdf', '.PDF')):
                        await self._extract_knowledge_from_pdf_url(resource_value)
                    else:
                        await self._extract_knowledge_from_generic_url(resource_value)
                        
                    # Update processed URLs in state
                    self.current_state["processed_urls"].append({
                        "url": resource_value,
                        "source": "user_feedback",
                        "status": "processed"
                    })
                    self.current_state["resources_processed"] += 1
                    
                except Exception as e:
                    logger.error(f"Error processing URL {resource_value}: {e}")
            
            elif resource_type == "file":
                # Process a local file path
                logger.info(f"Processing file from feedback: {resource_value}")
                try:
                    await self._process_input_files([resource_value])
                except Exception as e:
                    logger.error(f"Error processing file {resource_value}: {e}")
            
            elif resource_type == "description":
                # Process a text description
                logger.info(f"Processing text description from feedback")
                try:
                    await self._process_text_content(resource_value, "user_feedback")
                except Exception as e:
                    logger.error(f"Error processing text description: {e}")
    
    async def _incorporate_focus_feedback(self, feedback: UserFeedback) -> None:
        """Incorporate feedback about research focus areas."""
        logger.info(f"Incorporating focus feedback: {feedback.id}")
        
        # Extract focus information
        focus_title = f"Focus from feedback {feedback.id[:8]}"
        focus_description = feedback.text
        
        # Create a new research focus
        focus = ResearchFocus(
            research_goal_id=self.current_research_goal.id,
            user_id=feedback.user_id,
            title=focus_title,
            description=focus_description,
            priority=feedback.priority / 5.0 if feedback.priority else 1.0,
            active=True
        )
        
        # Extract keywords
        try:
            keywords = await self.supervisor.extract_keywords(focus_description)
            if keywords:
                focus.keywords = keywords
        except Exception as e:
            logger.warning(f"Could not extract keywords: {e}")
            
        # Save to database
        self.db.research_focus.save(focus)
        logger.info(f"Created research focus {focus.id} from feedback")
        
        # Update focus areas count
        self.current_state["focus_areas_count"] += 1
    
    async def _incorporate_experimental_results(self, feedback: UserFeedback) -> None:
        """Incorporate experimental results from user feedback."""
        logger.info(f"Incorporating experimental results: {feedback.id}")
        
        # This would process experimental results provided by the user
        # and update the related hypotheses or protocols
        
        if feedback.hypothesis_id:
            # Update the hypothesis based on experimental results
            hypothesis = self.db.hypotheses.get(feedback.hypothesis_id)
            if hypothesis:
                # Add experimental results to metadata
                if "experimental_results" not in hypothesis.metadata:
                    hypothesis.metadata["experimental_results"] = []
                    
                hypothesis.metadata["experimental_results"].append({
                    "feedback_id": feedback.id,
                    "text": feedback.text,
                    "timestamp": datetime.now().isoformat(),
                    "user_id": feedback.user_id
                })
                
                # Update hypothesis status based on results
                if "verdict" in feedback.metadata:
                    verdict = feedback.metadata["verdict"]
                    if verdict == "confirmed":
                        hypothesis.status = HypothesisStatus.ACCEPTED
                    elif verdict == "rejected":
                        hypothesis.status = HypothesisStatus.REJECTED
                
                # Save updated hypothesis
                self.db.hypotheses.save(hypothesis)
                logger.info(f"Updated hypothesis {hypothesis.id} with experimental results")
                
                # This could trigger specific follow-up actions based on the results
    
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
        """Start the system in interactive mode with free-form commands."""
        print("Welcome to Raul Co-Scientist interactive mode!")
        print("Enter a research goal to begin, or type 'help' for available commands.")
        
        # Default user ID for the current session
        default_user_id = str(uuid.uuid4())
        
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
                
            elif user_input == "hypotheses":
                # List all hypotheses
                if not self.current_research_goal:
                    print("No research goal set. Please set a goal first.")
                    continue
                
                # Get all hypotheses sorted by Elo rating
                all_hypotheses = self.db.hypotheses.get_all()
                sorted_hypotheses = sorted(all_hypotheses, key=lambda h: h.elo_rating if h.elo_rating is not None else 1200, reverse=True)
                
                if not sorted_hypotheses:
                    print("No hypotheses have been generated yet.")
                    continue
                
                print("\n============ HYPOTHESES ============")
                for i, hypothesis in enumerate(sorted_hypotheses, 1):
                    # Get source indicator
                    source_indicator = {
                        HypothesisSource.USER: "👤",
                        HypothesisSource.SYSTEM: "🤖",
                        HypothesisSource.EVOLVED: "🧬",
                        HypothesisSource.COMBINED: "🔄"
                    }.get(hypothesis.source, "")
                    
                    # Format the output
                    print(f"{i}. {source_indicator} {hypothesis.title} (ID: {hypothesis.id[:8]})")
                    print(f"   Summary: {hypothesis.summary[:100]}...")
                    print(f"   Elo Rating: {hypothesis.elo_rating}")
                    
                    # Show scores if available
                    scores = []
                    if hypothesis.novelty_score is not None:
                        scores.append(f"Novelty: {hypothesis.novelty_score:.1f}")
                    if hypothesis.correctness_score is not None:
                        scores.append(f"Correctness: {hypothesis.correctness_score:.1f}")
                    if hypothesis.testability_score is not None:
                        scores.append(f"Testability: {hypothesis.testability_score:.1f}")
                    
                    if scores:
                        print(f"   Scores: {', '.join(scores)}")
                    
                    # Show citation count
                    citation_count = len(hypothesis.citations) if hypothesis.citations else 0
                    print(f"   Citations: {citation_count}")
                    print()
                
                print("To view a specific hypothesis, use 'hypothesis:ID' (using the ID shown above)")
                
            elif user_input.startswith("hypothesis:"):
                # Get ID from input
                hypothesis_id = user_input[11:].strip()
                
                # First try to match by short ID prefix
                all_hypotheses = self.db.hypotheses.get_all()
                matching_hypotheses = [h for h in all_hypotheses if h.id.startswith(hypothesis_id)]
                
                if not matching_hypotheses:
                    print(f"No hypothesis found with ID starting with {hypothesis_id}")
                    continue
                
                # Use the first match
                hypothesis = matching_hypotheses[0]
                
                # Get reviews for this hypothesis
                reviews = [r for r in self.db.reviews.get_all() if r.hypothesis_id == hypothesis.id]
                
                # Display detailed hypothesis
                print("\n============ HYPOTHESIS DETAILS ============")
                print(f"Title: {hypothesis.title}")
                print(f"ID: {hypothesis.id}")
                print(f"Created: {hypothesis.created_at.strftime('%Y-%m-%d %H:%M')}")
                print(f"Source: {hypothesis.source.value}")
                print(f"Creator: {hypothesis.creator}")
                print(f"Status: {hypothesis.status.value}")
                print(f"\nSummary: {hypothesis.summary}")
                print(f"\nDescription:\n{hypothesis.description}")
                
                if hypothesis.supporting_evidence:
                    print("\nSupporting Evidence:")
                    for i, evidence in enumerate(hypothesis.supporting_evidence, 1):
                        print(f"  {i}. {evidence}")
                
                if hasattr(hypothesis, 'experimental_approaches') and hypothesis.experimental_approaches:
                    print("\nExperimental Approaches:")
                    for i, approach in enumerate(hypothesis.experimental_approaches, 1):
                        print(f"  {i}. {approach}")
                        
                if hasattr(hypothesis, 'predictions') and hypothesis.predictions:
                    print("\nPredictions:")
                    for i, prediction in enumerate(hypothesis.predictions, 1):
                        print(f"  {i}. {prediction}")
                
                if hypothesis.citations:
                    print("\nCitations:")
                    for i, citation in enumerate(hypothesis.citations, 1):
                        # Handle both Citation objects and citation IDs
                        if isinstance(citation, str):
                            citation_obj = self.db.citations.get(citation)
                            if citation_obj:
                                title = citation_obj.title
                                authors = ", ".join(citation_obj.authors[:3]) if citation_obj.authors else "Unknown"
                                year = citation_obj.year if citation_obj.year else "N/A"
                                print(f"  {i}. {title} ({authors}, {year})")
                            else:
                                print(f"  {i}. Citation ID: {citation}")
                        else:
                            title = citation.title
                            authors = ", ".join(citation.authors[:3]) if citation.authors else "Unknown"
                            year = citation.year if citation.year else "N/A"
                            print(f"  {i}. {title} ({authors}, {year})")
                
                if reviews:
                    print("\nReviews:")
                    for i, review in enumerate(reviews, 1):
                        print(f"  {i}. {review.review_type.capitalize()} Review by {review.reviewer}")
                        if review.overall_score is not None:
                            print(f"     Score: {review.overall_score:.1f}/10")
                        print(f"     {review.text[:100]}...")
                
                print("\nOptions for this hypothesis:")
                print("  feedback:ID Your feedback here - Provide feedback on this hypothesis")
                print("  evolve:ID - Request evolution of this hypothesis")
                print("  protocol:ID - Generate experimental protocol for this hypothesis")
                
            elif user_input.startswith("feedback:"):
                # Extract the parts: "feedback:ID Your feedback text"
                parts = user_input[9:].strip().split(" ", 1)
                
                if len(parts) < 2:
                    print("Please provide both an ID and feedback text. Format: feedback:ID Your feedback here")
                    continue
                
                hypothesis_id = parts[0]
                feedback_text = parts[1]
                
                # Find the hypothesis
                all_hypotheses = self.db.hypotheses.get_all()
                matching_hypotheses = [h for h in all_hypotheses if h.id.startswith(hypothesis_id)]
                
                if not matching_hypotheses:
                    print(f"No hypothesis found with ID starting with {hypothesis_id}")
                    continue
                
                # Use the first match
                hypothesis = matching_hypotheses[0]
                
                # Create feedback
                feedback = UserFeedback(
                    research_goal_id=self.current_research_goal.id,
                    hypothesis_id=hypothesis.id,
                    user_id=default_user_id,
                    feedback_type="critique",
                    text=feedback_text,
                    requires_action=True
                )
                
                # Save feedback
                self.db.user_feedback.save(feedback)
                
                print(f"Feedback recorded for hypothesis: {hypothesis.title}")
                print("This feedback will be used in the next iteration to improve the hypothesis.")
                
            elif user_input.startswith("evolve:"):
                # Extract hypothesis ID
                hypothesis_id = user_input[7:].strip()
                
                # Find the hypothesis
                all_hypotheses = self.db.hypotheses.get_all()
                matching_hypotheses = [h for h in all_hypotheses if h.id.startswith(hypothesis_id)]
                
                if not matching_hypotheses:
                    print(f"No hypothesis found with ID starting with {hypothesis_id}")
                    continue
                
                # Use the first match
                hypothesis = matching_hypotheses[0]
                
                print(f"Evolving hypothesis: {hypothesis.title}")
                
                # Manually trigger hypothesis evolution
                try:
                    improved_hypothesis = await self.evolution.improve_hypothesis(
                        hypothesis,
                        self.current_research_goal,
                        strategy="human_directed"
                    )
                    
                    # Save the new hypothesis
                    if improved_hypothesis:
                        improved_hypothesis.parent_hypotheses = [hypothesis.id]
                        improved_hypothesis.source = HypothesisSource.EVOLVED
                        self.db.hypotheses.save(improved_hypothesis)
                        
                        print(f"Successfully evolved hypothesis into: {improved_hypothesis.title}")
                        print(f"New hypothesis ID: {improved_hypothesis.id[:8]}")
                    else:
                        print("Failed to evolve hypothesis.")
                except Exception as e:
                    print(f"Error evolving hypothesis: {e}")
                
            elif user_input.startswith("focus:"):
                # Add a research focus area
                focus_text = user_input[6:].strip()
                
                if not focus_text:
                    print("Please provide a description for the research focus area.")
                    continue
                
                # Create a new research focus
                focus = ResearchFocus(
                    research_goal_id=self.current_research_goal.id,
                    user_id=default_user_id,
                    title=f"Focus area: {focus_text[:30]}",
                    description=focus_text,
                    priority=1.0,
                    active=True
                )
                
                # Extract keywords
                try:
                    # Extract keywords using LLM
                    if hasattr(self, 'supervisor'):
                        keywords = await self.supervisor.extract_keywords(focus_text)
                        if keywords:
                            focus.keywords = keywords
                except Exception as e:
                    print(f"Warning: Could not extract keywords: {e}")
                
                # Save to database
                self.db.research_focus.save(focus)
                
                print(f"Added new research focus area: {focus.title}")
                print("This focus area will guide future hypothesis generation and evolution.")
                
            elif user_input == "focus-areas":
                # List active research focus areas
                if not self.current_research_goal:
                    print("No research goal set. Please set a goal first.")
                    continue
                
                # Get all focus areas for this research goal
                all_focus = self.db.research_focus.get_all()
                relevant_focus = [f for f in all_focus if f.research_goal_id == self.current_research_goal.id and f.active]
                
                if not relevant_focus:
                    print("No active research focus areas. Add one with 'focus: Your focus area description'")
                    continue
                
                print("\n============ ACTIVE RESEARCH FOCUS AREAS ============")
                for i, focus in enumerate(relevant_focus, 1):
                    print(f"{i}. {focus.title}")
                    print(f"   Description: {focus.description}")
                    if focus.keywords:
                        print(f"   Keywords: {', '.join(focus.keywords)}")
                    print(f"   Priority: {focus.priority}")
                    print()
                
            elif user_input.startswith("resource:"):
                # Add an external resource
                resource_text = user_input[9:].strip()
                
                if not resource_text:
                    print("Please provide details for the resource.")
                    continue
                
                # Check if it looks like a URL
                is_url = resource_text.startswith(("http://", "https://"))
                
                # Create appropriate feedback type
                feedback = UserFeedback(
                    research_goal_id=self.current_research_goal.id,
                    user_id=default_user_id,
                    feedback_type="resource",
                    text=f"Please consider this resource in your analysis: {resource_text}",
                    resources=[{"url" if is_url else "description": resource_text}],
                    requires_action=True
                )
                
                # Save to database
                self.db.user_feedback.save(feedback)
                
                print(f"Added new resource: {resource_text[:50]}...")
                print("This resource will be considered in future iterations.")
                
                # If it's a URL and looks like a PDF, ask if they want to extract it
                if is_url and resource_text.endswith((".pdf", ".PDF")):
                    print("\nThis appears to be a PDF link. Would you like to extract knowledge from it? (y/n)")
                    extract_response = input("> ")
                    
                    if extract_response.lower() in ["y", "yes"]:
                        print("Attempting to extract knowledge from PDF...")
                        try:
                            if hasattr(self, 'evolution') and hasattr(self.evolution, 'paper_extraction'):
                                await self.evolution.paper_extraction.extract_from_url(resource_text)
                                print("Successfully extracted knowledge from PDF and added to knowledge graph.")
                            else:
                                print("Paper extraction system not available.")
                        except Exception as e:
                            print(f"Error extracting from PDF: {e}")
                
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
                
            elif user_input.startswith("protocol:"):
                # Generate a protocol for a specific hypothesis
                hypothesis_id = user_input[9:].strip()
                
                # Find the hypothesis
                all_hypotheses = self.db.hypotheses.get_all()
                matching_hypotheses = [h for h in all_hypotheses if h.id.startswith(hypothesis_id)]
                
                if not matching_hypotheses:
                    print(f"No hypothesis found with ID starting with {hypothesis_id}")
                    continue
                
                # Use the first match
                hypothesis = matching_hypotheses[0]
                
                print(f"Generating experimental protocol for hypothesis: {hypothesis.title}")
                
                # Generate the protocol
                try:
                    protocol = await self._generate_protocol_for_hypothesis(hypothesis)
                    if protocol:
                        print(f"Generated protocol: {protocol.title}")
                        print(f"Protocol ID: {protocol.id[:8]}")
                    else:
                        print("Failed to generate protocol.")
                except Exception as e:
                    print(f"Error generating protocol: {e}")
                
            elif user_input == "generate-protocol":
                # Generate a new protocol for a top hypothesis
                if not self.current_research_goal:
                    print("No research goal set. Please set a goal first.")
                    continue
                
                print("Generating experimental protocol for a top hypothesis...")
                await self._generate_protocol()
                print("Protocol generation completed.")
                
            elif user_input.startswith("search:"):
                # Search across scientific databases
                query = user_input[7:].strip()
                if not query:
                    print("Please provide a search query after 'search:'")
                    continue
                    
                print(f"Searching scientific databases for: {query}")
                
                try:
                    # Use cross-domain synthesizer to detect relevant domains
                    domains = []
                    if hasattr(self, 'cross_domain_synthesizer'):
                        domain_info = self.cross_domain_synthesizer.detect_research_domains(query)
                        domains = [domain for domain, score in domain_info if score > 0.4][:3]
                        
                        print(f"Detected relevant domains: {', '.join(domains)}")
                    
                    # Initialize domain knowledge if needed
                    if hasattr(self, 'domain_knowledge') and not self.domain_knowledge.initialized:
                        print("Initializing domain knowledge...")
                        await self.domain_knowledge.initialize(domains=domains)
                    
                    # Search across domains
                    results = await self.domain_knowledge.search(query, domains=domains, limit=3)
                    
                    # Display results
                    print("\n============ SEARCH RESULTS ============")
                    for domain, domain_results in results.items():
                        print(f"\n{domain.upper()} DOMAIN:")
                        if not domain_results:
                            print("  No results found")
                            continue
                            
                        for i, result in enumerate(domain_results, 1):
                            print(f"  {i}. {result.get('title', 'Untitled')}")
                            if result.get('authors'):
                                print(f"     Authors: {', '.join(result['authors'][:3])}")
                            if result.get('journal'):
                                print(f"     Journal: {result['journal']}")
                            if result.get('year'):
                                print(f"     Year: {result['year']}")
                            if result.get('url'):
                                print(f"     URL: {result['url']}")
                            print()
                    
                    # Ask if the user wants to add any of these as resources
                    print("\nWould you like to add any of these results as resources? (Enter numbers separated by commas, or 'n' to skip)")
                    add_response = input("> ")
                    
                    if add_response.lower() not in ["n", "no", ""]:
                        try:
                            # Parse selected indices
                            selected_indices = [int(idx.strip()) - 1 for idx in add_response.split(",")]
                            
                            # Flatten results from all domains
                            all_results = []
                            for domain_results in results.values():
                                all_results.extend(domain_results)
                            
                            # Add selected results as resources
                            for idx in selected_indices:
                                if 0 <= idx < len(all_results):
                                    result = all_results[idx]
                                    resource_url = result.get('url')
                                    if resource_url:
                                        resource_feedback = UserFeedback(
                                            research_goal_id=self.current_research_goal.id,
                                            user_id=default_user_id,
                                            feedback_type="resource",
                                            text=f"Please consider this resource: {result.get('title', 'Untitled article')}",
                                            resources=[{"url": resource_url, "title": result.get('title'), "type": "paper"}],
                                            requires_action=True
                                        )
                                        self.db.user_feedback.save(resource_feedback)
                                        print(f"Added resource: {result.get('title')}")
                        except Exception as e:
                            print(f"Error adding resources: {e}")
                    
                except Exception as e:
                    print(f"Error searching scientific databases: {e}")
                    
            elif user_input.startswith("synthesize:"):
                # Synthesize knowledge across domains
                query = user_input[11:].strip()
                if not query:
                    print("Please provide a query after 'synthesize:'")
                    continue
                    
                print(f"Synthesizing knowledge across domains for: {query}")
                
                try:
                    if not hasattr(self, 'cross_domain_synthesizer'):
                        print("Cross-domain synthesizer not available")
                        continue
                        
                    # Use cross-domain synthesizer to detect domains and synthesize knowledge
                    domain_info = self.cross_domain_synthesizer.detect_research_domains(query)
                    domains = [domain for domain, score in domain_info if score > 0.4][:3]
                    
                    print(f"Detected relevant domains: {', '.join(domains)}")
                    
                    # Initialize domain knowledge if needed
                    if hasattr(self, 'domain_knowledge') and not self.domain_knowledge.initialized:
                        print("Initializing domain knowledge...")
                        await self.domain_knowledge.initialize(domains=domains)
                    
                    # Perform synthesis
                    print("Gathering and synthesizing information across domains...")
                    synthesis = await self.cross_domain_synthesizer.synthesize_multi_domain_knowledge(
                        query, max_entities_per_domain=2, max_related_per_entity=2
                    )
                    
                    # Format and display highlights
                    if synthesis:
                        highlights = self.cross_domain_synthesizer.format_synthesis_highlights(synthesis)
                        print("\n" + highlights)
                        
                        # Ask if the user wants to add this as context
                        print("\nWould you like to add this synthesis as context for future hypotheses? (y/n)")
                        add_response = input("> ")
                        
                        if add_response.lower() in ["y", "yes"]:
                            context_feedback = UserFeedback(
                                research_goal_id=self.current_research_goal.id,
                                user_id=default_user_id,
                                feedback_type="context",
                                text=f"Synthesis of knowledge for query: {query}\n\n{highlights}",
                                requires_action=True
                            )
                            self.db.user_feedback.save(context_feedback)
                            print("Added synthesis as research context.")
                    else:
                        print("No synthesis results available.")
                    
                except Exception as e:
                    print(f"Error synthesizing knowledge: {e}")
                
            elif user_input.startswith("add-hypothesis:"):
                # Add a user hypothesis
                hypothesis_text = user_input[15:].strip()
                
                if not hypothesis_text:
                    print("Please provide a hypothesis after 'add-hypothesis:'")
                    continue
                
                print("Adding your hypothesis. Let's add some details...")
                print("\nPlease provide a title for your hypothesis:")
                title = input("> ")
                
                print("\nPlease provide a brief summary:")
                summary = input("> ")
                
                # Create the hypothesis
                hypothesis = Hypothesis(
                    title=title,
                    description=hypothesis_text,
                    summary=summary,
                    creator="user",
                    supporting_evidence=[],
                    source=HypothesisSource.USER,
                    status=HypothesisStatus.GENERATED,
                    user_id=default_user_id
                )
                
                # Save to database
                self.db.hypotheses.save(hypothesis)
                
                print(f"Added your hypothesis with ID: {hypothesis.id[:8]}")
                print("This hypothesis will be evaluated in the next iteration.")
                
            elif user_input == "feedback":
                # List all user feedback
                if not self.current_research_goal:
                    print("No research goal set. Please set a goal first.")
                    continue
                
                # Get all feedback for this research goal
                all_feedback = self.db.user_feedback.get_all()
                relevant_feedback = [f for f in all_feedback if f.research_goal_id == self.current_research_goal.id]
                
                if not relevant_feedback:
                    print("No feedback has been provided yet.")
                    continue
                
                print("\n============ USER FEEDBACK ============")
                for i, feedback in enumerate(relevant_feedback, 1):
                    print(f"{i}. Type: {feedback.feedback_type}")
                    print(f"   Created: {feedback.created_at.strftime('%Y-%m-%d %H:%M')}")
                    print(f"   {feedback.text[:100]}...")
                    
                    if feedback.resources:
                        print(f"   Resources: {len(feedback.resources)}")
                    
                    if feedback.requires_action:
                        action_status = "Pending" if not feedback.action_taken else "Completed"
                        print(f"   Action status: {action_status}")
                    
                    print()
                
            else:
                print("Unknown command. Type 'help' for a list of commands.")
    
    async def start_structured_interactive_mode(self) -> None:
        """
        Start the system in a structured, step-by-step interactive mode that follows
        the specified workflow in ETHOS.md:
        
        1. Set research goal
        2. Initial iteration to generate hypotheses
        3. Present results and get feedback
        4. Incorporate feedback and run another iteration
        5. Repeat steps 3-4 until satisfied
        """
        # Import the user output formatter
        from ..utils.logger import user_output
        
        print(user_output.section_header("RAUL CO-SCIENTIST", width=70))
        print(user_output.info("Welcome to the structured interactive mode!"))
        print("\nThis mode will guide you through the complete research workflow:")
        print("  1. " + user_output.highlight("Set your research goal"))
        print("  2. " + user_output.highlight("Generate initial hypotheses and research directions"))
        print("  3. " + user_output.highlight("Provide feedback and refine the research"))
        print("  4. " + user_output.highlight("Iterate to improve hypotheses based on your feedback"))
        print("\nYou can exit at any time by typing 'exit' or 'quit'.")
        
        # Step 1: Set Research Goal
        if not self.current_research_goal:
            print(user_output.section_header("STEP 1: SET RESEARCH GOAL", width=70))
            print("Please enter your research goal or question:")
            
            while True:
                research_goal_text = input("> ")
                
                if research_goal_text.lower() in ["exit", "quit"]:
                    print("Exiting interactive mode.")
                    return
                
                if not research_goal_text or len(research_goal_text) < 10:
                    print("Please enter a meaningful research goal (at least 10 characters).")
                    continue
                
                # Ask for input files
                print("\nWould you like to provide any input files (local documents, PDFs, data files)? (y/n)")
                files_response = input("> ")
                
                input_files = []
                if files_response.lower() == 'y':
                    print("Enter the paths to input files (one per line, empty line to finish):")
                    while True:
                        file_path = input("> ")
                        if not file_path:
                            break
                        input_files.append(file_path)
                    
                    if input_files:
                        print(f"Added {len(input_files)} input files.")
                
                # Ask for URLs
                print("\nWould you like to provide any URLs (web pages, online papers, etc.)? (y/n)")
                urls_response = input("> ")
                
                input_urls = []
                if urls_response.lower() == 'y':
                    print("Enter URLs (one per line, empty line to finish):")
                    while True:
                        url = input("> ")
                        if not url:
                            break
                        input_urls.append(url)
                    
                    if input_urls:
                        print(user_output.success(f"Added {len(input_urls)} URLs."))
                
                print(f"\n{user_output.info('Analyzing research goal:')} {research_goal_text[:100]}...")
                print(user_output.info("This may take a moment as we process your inputs..."))
                
                try:
                    await self.analyze_research_goal(
                        research_goal_text, 
                        input_files=input_files if input_files else None,
                        urls=input_urls if input_urls else None
                    )
                    print(user_output.success(f"Research goal set with ID: {self.current_research_goal.id}"))
                    break
                except Exception as e:
                    print(user_output.error(f"Error setting research goal: {e}"))
                    print("Please try again with a different research goal:")
                    
        # If we're resuming an existing goal, print it for reference
        else:
            print(user_output.section_header("RESUMING EXISTING RESEARCH GOAL", width=70))
            print(f"Current research goal: {self.current_research_goal.text[:150]}...")
            print(f"Goal ID: {user_output.highlight(self.current_research_goal.id)}")
        
        # Main loop for iterations
        while True:
            if self.session_state == SessionState.WAITING_FOR_FEEDBACK:
                # We're waiting for feedback after completing an iteration
                print(user_output.section_header("FEEDBACK REQUIRED", width=70))
                print(user_output.info("The system has completed an iteration and is waiting for your feedback."))
                print("Please review the current hypotheses and provide guidance.")
                
                # Show current top hypotheses
                top_hypotheses = []
                for h_id in self.current_state["top_hypotheses"]:
                    hypothesis = self.db.hypotheses.get(h_id)
                    if hypothesis:
                        top_hypotheses.append(hypothesis)
                
                if top_hypotheses:
                    print(user_output.subsection_header("TOP HYPOTHESES"))
                    for i, h in enumerate(top_hypotheses):
                        print(f"\n{i+1}. {user_output.highlight(h.title)}")
                        print(f"   Summary: {h.summary}")
                else:
                    print("\n" + user_output.error("No hypotheses generated yet."))
                
                # Get research overview
                overview = None
                if self.current_state.get("last_research_overview_id"):
                    overview = self.db.research_overviews.get(self.current_state["last_research_overview_id"])
                
                if overview:
                    print(user_output.subsection_header("RESEARCH OVERVIEW"))
                    print(f"Title: {user_output.highlight(overview.title)}")
                    print(f"Summary: {overview.summary[:200]}...")
                    print("\nResearch Areas:")
                    for i, area in enumerate(overview.research_areas[:3]):  # Show just a few areas
                        area_title = area.get('title', 'Unnamed Area')
                        print(f"  {i+1}. {user_output.highlight(area_title)}")
                
                # Feedback options
                print(user_output.subsection_header("FEEDBACK OPTIONS"))
                print("Please choose a feedback option:")
                print("1. " + user_output.highlight("Provide general feedback") + " on the research direction")
                print("2. " + user_output.highlight("Add a specific research focus area") + " to guide exploration")
                print("3. " + user_output.highlight("Provide feedback on a specific hypothesis"))
                print("4. " + user_output.highlight("Add a resource") + " (paper, URL, or document)")
                print("5. " + user_output.highlight("Provide experimental results"))
                print("6. " + user_output.highlight("View full hypothesis details"))
                print("7. " + user_output.highlight("Continue without feedback"))
                print("8. " + user_output.highlight("Exit the system"))
                
                # Use a more robust input handling pattern
                while True:
                    option = input("\nEnter your choice (1-8): ").strip()
                    
                    if option in ["1", "2", "3", "4", "5", "6", "7", "8"]:
                        break
                    else:
                        print(user_output.error("Invalid option. Please enter a number between 1 and 8."))
                        # Add a small delay to prevent UI overlap
                        await asyncio.sleep(0.1)
                
                if option == "1":
                    # General feedback
                    print("\nPlease provide your general feedback on the research direction:")
                    feedback_text = input("> ")
                    
                    if not feedback_text:
                        print("No feedback provided.")
                        continue
                    
                    # Create feedback object
                    feedback = UserFeedback(
                        research_goal_id=self.current_research_goal.id,
                        user_id=self.current_user_id,
                        feedback_type="general",
                        text=feedback_text,
                        requires_action=True
                    )
                    self.db.user_feedback.save(feedback)
                    print(user_output.success("Thank you for your feedback. It will be incorporated in the next iteration."))
                    # Change state to allow running the next iteration
                    self.session_state = SessionState.PROCESSING_FEEDBACK
                
                elif option == "2":
                    # Research focus
                    print("\nPlease describe a specific focus area for the research:")
                    focus_text = input("> ")
                    
                    if not focus_text:
                        print("No focus area provided.")
                        continue
                    
                    # Create feedback for focus
                    feedback = UserFeedback(
                        research_goal_id=self.current_research_goal.id,
                        user_id=self.current_user_id,
                        feedback_type="focus",
                        text=focus_text,
                        requires_action=True
                    )
                    self.db.user_feedback.save(feedback)
                    print(user_output.success("Focus area added. It will be incorporated in the next iteration."))
                    # Change state to allow running the next iteration
                    self.session_state = SessionState.PROCESSING_FEEDBACK
                
                elif option == "3":
                    # Hypothesis feedback
                    if not top_hypotheses:
                        print(user_output.error("No hypotheses available to provide feedback on."))
                        continue
                    
                    print("\nWhich hypothesis would you like to provide feedback on? (enter number)")
                    
                    # Use a more robust input handling pattern for hypothesis selection
                    selected_hypothesis = None
                    while not selected_hypothesis:
                        try:
                            h_input = input("> ").strip()
                            h_num = int(h_input)
                            
                            if h_num < 1 or h_num > len(top_hypotheses):
                                print(user_output.error(f"Please enter a number between 1 and {len(top_hypotheses)}."))
                                continue
                                
                            selected_hypothesis = top_hypotheses[h_num - 1]
                            
                        except ValueError:
                            print(user_output.error("Please enter a valid number."))
                            # Add a small delay to prevent UI overlap
                            await asyncio.sleep(0.1)
                            continue
                    
                    print(f"\nProviding feedback for: {user_output.highlight(selected_hypothesis.title)}")
                    print(f"Description: {selected_hypothesis.description[:200]}...")
                    
                    print("\nEnter your feedback on this hypothesis:")
                    h_feedback = input("> ")
                    
                    if not h_feedback:
                        print("No feedback provided.")
                        continue
                    
                    # Create feedback for hypothesis
                    feedback = UserFeedback(
                        research_goal_id=self.current_research_goal.id,
                        user_id=self.current_user_id,
                        feedback_type="hypothesis",
                        hypothesis_id=selected_hypothesis.id,
                        text=h_feedback,
                        requires_action=True
                    )
                    self.db.user_feedback.save(feedback)
                    print(user_output.success("Hypothesis feedback added. It will be incorporated in the next iteration."))
                    # Change state to allow running the next iteration
                    self.session_state = SessionState.PROCESSING_FEEDBACK
                    
                elif option == "4":
                    # Add resource
                    print("\nWhat type of resource would you like to add?")
                    print("1. " + user_output.highlight("URL") + " (web page, online paper)")
                    print("2. " + user_output.highlight("Local file") + " (PDF, text, data)")
                    print("3. " + user_output.highlight("Text description"))
                    
                    # Use a more robust input handling pattern
                    while True:
                        resource_type = input("> ").strip()
                        
                        if resource_type in ["1", "2", "3"]:
                            break
                        else:
                            print(user_output.error("Invalid option. Please enter 1, 2, or 3."))
                            # Add a small delay to prevent UI overlap
                            await asyncio.sleep(0.1)
                    
                    if resource_type == "1":
                        print("\nEnter the URL:")
                        url = input("> ")
                        
                        if not url:
                            print("No URL provided.")
                            continue
                        
                        # Create feedback with URL resource
                        feedback = UserFeedback(
                            research_goal_id=self.current_research_goal.id,
                            user_id=self.current_user_id,
                            feedback_type="resource",
                            text=f"Please analyze this URL: {url}",
                            resources=[{"url": url}],
                            requires_action=True
                        )
                        self.db.user_feedback.save(feedback)
                        print(user_output.success(f"URL resource added: {url}"))
                        # Change state to allow running the next iteration
                        self.session_state = SessionState.PROCESSING_FEEDBACK
                    
                    elif resource_type == "2":
                        print("\nEnter the path to the local file:")
                        file_path = input("> ")
                        
                        if not file_path:
                            print("No file path provided.")
                            continue
                        
                        # Check if file exists
                        if not os.path.exists(file_path):
                            print(f"File not found: {file_path}")
                            continue
                        
                        # Create feedback with file resource
                        feedback = UserFeedback(
                            research_goal_id=self.current_research_goal.id,
                            user_id=self.current_user_id,
                            feedback_type="resource",
                            text=f"Please analyze this file: {file_path}",
                            resources=[{"file": file_path}],
                            requires_action=True
                        )
                        self.db.user_feedback.save(feedback)
                        print(user_output.success(f"File resource added: {file_path}"))
                        # Change state to allow running the next iteration
                        self.session_state = SessionState.PROCESSING_FEEDBACK
                    
                    elif resource_type == "3":
                        print("\nEnter your text description:")
                        description = input("> ")
                        
                        if not description:
                            print("No description provided.")
                            continue
                        
                        # Create feedback with text description
                        feedback = UserFeedback(
                            research_goal_id=self.current_research_goal.id,
                            user_id=self.current_user_id,
                            feedback_type="resource",
                            text="Please consider this information in your analysis.",
                            resources=[{"description": description}],
                            requires_action=True
                        )
                        self.db.user_feedback.save(feedback)
                        print(user_output.success("Text description added."))
                        # Change state to allow running the next iteration
                        self.session_state = SessionState.PROCESSING_FEEDBACK
                    
                    else:
                        print("Invalid option.")
                        continue
                
                elif option == "5":
                    # Experimental results
                    if not top_hypotheses:
                        print(user_output.error("No hypotheses available to provide experimental results for."))
                        continue
                    
                    print("\nWhich hypothesis would you like to provide experimental results for? (enter number)")
                    
                    # Use a more robust input handling pattern for hypothesis selection
                    selected_hypothesis = None
                    while not selected_hypothesis:
                        try:
                            h_input = input("> ").strip()
                            h_num = int(h_input)
                            
                            if h_num < 1 or h_num > len(top_hypotheses):
                                print(user_output.error(f"Please enter a number between 1 and {len(top_hypotheses)}."))
                                continue
                                
                            selected_hypothesis = top_hypotheses[h_num - 1]
                            
                        except ValueError:
                            print(user_output.error("Please enter a valid number."))
                            # Add a small delay to prevent UI overlap
                            await asyncio.sleep(0.1)
                            continue
                    
                    print(f"\nProviding experimental results for: {user_output.highlight(selected_hypothesis.title)}")
                    
                    print("\nEnter your experimental results:")
                    exp_results = input("> ")
                    
                    if not exp_results:
                        print("No results provided.")
                        continue
                    
                    print("\nDid the experiment confirm or reject the hypothesis?")
                    print("1. " + user_output.highlight("Confirm"))
                    print("2. " + user_output.highlight("Reject"))
                    print("3. " + user_output.highlight("Inconclusive"))
                    
                    # Use a more robust input handling pattern
                    while True:
                        verdict_input = input("> ").strip()
                        
                        if verdict_input in ["1", "2", "3"]:
                            break
                        else:
                            print(user_output.error("Invalid option. Please enter 1, 2, or 3."))
                            # Add a small delay to prevent UI overlap
                            await asyncio.sleep(0.1)
                    
                    verdict = None
                    if verdict_input == "1":
                        verdict = "confirmed"
                    elif verdict_input == "2":
                        verdict = "rejected"
                    elif verdict_input == "3":
                        verdict = "inconclusive"
                    
                    # Create feedback for experimental results
                    feedback = UserFeedback(
                        research_goal_id=self.current_research_goal.id,
                        user_id=self.current_user_id,
                        feedback_type="experimental",
                        hypothesis_id=selected_hypothesis.id,
                        text=exp_results,
                        requires_action=True,
                        metadata={"verdict": verdict} if verdict else {}
                    )
                    self.db.user_feedback.save(feedback)
                    print(user_output.success("Experimental results added. They will be incorporated in the next iteration."))
                    # Change state to allow running the next iteration
                    self.session_state = SessionState.PROCESSING_FEEDBACK
                    
                elif option == "6":
                    # View full hypothesis details
                    if not top_hypotheses:
                        print(user_output.error("No hypotheses available to view details of."))
                        continue
                    
                    print("\nWhich hypothesis would you like to view details of? (enter number)")
                    
                    # Use a more robust input handling pattern for hypothesis selection
                    selected_hypothesis = None
                    while not selected_hypothesis:
                        try:
                            h_input = input("> ").strip()
                            h_num = int(h_input)
                            
                            if h_num < 1 or h_num > len(top_hypotheses):
                                print(user_output.error(f"Please enter a number between 1 and {len(top_hypotheses)}."))
                                continue
                                
                            selected_hypothesis = top_hypotheses[h_num - 1]
                            
                        except ValueError:
                            print(user_output.error("Please enter a valid number."))
                            # Add a small delay to prevent UI overlap
                            await asyncio.sleep(0.1)
                            continue
                    
                    print(f"\nViewing details of: {user_output.highlight(selected_hypothesis.title)}")
                    print(f"ID: {selected_hypothesis.id}")
                    print(f"Created: {selected_hypothesis.created_at.strftime('%Y-%m-%d %H:%M')}")
                    print(f"Source: {selected_hypothesis.source.value}")
                    print(f"Creator: {selected_hypothesis.creator}")
                    print(f"Status: {selected_hypothesis.status.value}")
                    print(f"\nSummary: {selected_hypothesis.summary}")
                    print(f"\nDescription:\n{selected_hypothesis.description}")
                    
                    if selected_hypothesis.supporting_evidence:
                        print("\nSupporting Evidence:")
                        for i, evidence in enumerate(selected_hypothesis.supporting_evidence, 1):
                            print(f"  {i}. {evidence}")
                    
                    if hasattr(selected_hypothesis, 'experimental_approaches') and selected_hypothesis.experimental_approaches:
                        print("\nExperimental Approaches:")
                        for i, approach in enumerate(selected_hypothesis.experimental_approaches, 1):
                            print(f"  {i}. {approach}")
                            
                    if hasattr(selected_hypothesis, 'predictions') and selected_hypothesis.predictions:
                        print("\nPredictions:")
                        for i, prediction in enumerate(selected_hypothesis.predictions, 1):
                            print(f"  {i}. {prediction}")
                    
                    if selected_hypothesis.citations:
                        print("\nCitations:")
                        for i, citation in enumerate(selected_hypothesis.citations, 1):
                            # Handle both Citation objects and citation IDs
                            if isinstance(citation, str):
                                citation_obj = self.db.citations.get(citation)
                                if citation_obj:
                                    title = citation_obj.title
                                    authors = ", ".join(citation_obj.authors[:3]) if citation_obj.authors else "Unknown"
                                    year = citation_obj.year if citation_obj.year else "N/A"
                                    print(f"  {i}. {title} ({authors}, {year})")
                                else:
                                    print(f"  {i}. Citation ID: {citation}")
                            else:
                                title = citation.title
                                authors = ", ".join(citation.authors[:3]) if citation.authors else "Unknown"
                                year = citation.year if citation.year else "N/A"
                                print(f"  {i}. {title} ({authors}, {year})")
                    
                    # Get reviews for this hypothesis
                    reviews = self.db.get_reviews_for_hypothesis(selected_hypothesis.id)
                    
                    if reviews:
                        print("\nReviews:")
                        for i, review in enumerate(reviews, 1):
                            print(f"  {i}. {review.review_type.capitalize()} Review by {review.reviewer}")
                            if review.overall_score is not None:
                                print(f"     Score: {review.overall_score:.1f}/10")
                            print(f"     {review.text[:100]}...")
                    
                    print("\nOptions for this hypothesis:")
                    print("  feedback:ID Your feedback here - Provide feedback on this hypothesis")
                    print("  evolve:ID - Request evolution of this hypothesis")
                    print("  protocol:ID - Generate experimental protocol for this hypothesis")
                    
                elif option == "7":
                    # Continue without feedback - change state to allow running
                    print("\nContinuing without feedback...")
                    self.session_state = SessionState.GENERATING_HYPOTHESES
                
                elif option == "8":
                    print("\nExiting the system. Thanks for using Raul Co-Scientist!")
                    return
                
                else:
                    print("Invalid option. Please try again.")
                    continue
            
            # If we're not waiting for feedback, ask if user wants to run an iteration
            elif self.session_state != SessionState.WAITING_FOR_FEEDBACK:
                print(user_output.section_header("RESEARCH ITERATION", width=70))
                print("Would you like to run a research iteration? (y/n)")
                
                # Use a more robust input handling pattern
                while True:
                    response = input("> ").strip().lower()
                    
                    if response in ["n", "no", "exit", "quit"]:
                        print(user_output.info("Exiting research session."))
                        return
                    elif response in ["y", "yes"]:
                        break
                    else:
                        print(user_output.error("Please enter 'y' or 'n'."))
                
                print("\n" + user_output.info("Running research iteration..."))
                print(user_output.info("This may take a moment as we generate and evaluate hypotheses..."))
                
                try:
                    # Run one iteration
                    await self.run_iteration()
                    
                    # Show stats after iteration
                    print(user_output.section_header("ITERATION COMPLETED", width=70))
                    print(f"Total hypotheses: {user_output.highlight(str(self.current_state['num_hypotheses']))}")
                    print(f"Total reviews: {user_output.highlight(str(self.current_state['num_reviews']))}")
                    print(f"Iterations completed: {user_output.highlight(str(self.current_state['iterations_completed']))}")
                    
                    print("\n" + user_output.info("The system has completed an iteration and is now waiting for your feedback."))
                    
                    # Add a small delay to ensure UI elements don't overlap
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    print(user_output.error(f"Error running iteration: {e}"))
    
    def _print_help(self) -> None:
        """Print help information."""
        print("\n============ RAUL CO-SCIENTIST COMMANDS ============")
        
        print("\nResearch & Session Commands:")
        print("  goal: <text>       - Set a new research goal")
        print("  run [N]            - Run 1 or N iterations")
        print("  state              - Print the current system state")
        print("  overview           - Generate and print a research overview")
        
        print("\nHypothesis Management:")
        print("  hypotheses         - List all hypotheses, sorted by rating")
        print("  hypothesis:ID      - View detailed information about a specific hypothesis")
        print("  add-hypothesis:<text> - Add your own hypothesis") 
        print("  feedback:ID <text> - Provide feedback on a specific hypothesis")
        print("  evolve:ID          - Request evolution of a specific hypothesis")
        
        print("\nResearch Focus & Resources:")
        print("  focus: <text>      - Add a research focus area to guide exploration")
        print("  focus-areas        - List all active research focus areas")
        print("  resource: <url/text> - Add a resource (paper, URL, description)")
        print("  feedback           - List all feedback provided")
        
        print("\nProtocol Management:")
        print("  protocols          - List all experimental protocols")
        print("  protocol:ID        - Generate a protocol for a specific hypothesis")
        print("  generate-protocol  - Generate a protocol for a top hypothesis")
        
        print("\nKnowledge Search & Synthesis:")
        print("  search: <query>    - Search scientific databases across domains")
        print("  synthesize: <query> - Synthesize knowledge across scientific domains")
        
        print("\nSystem:")
        print("  help               - Show this help information")
        print("  exit or quit       - Exit the program")
        
        print("\nGuidelines for Effective Use:")
        print("  1. Start with a clear research goal using 'goal:' command")
        print("  2. Run iterations to generate initial hypotheses")
        print("  3. Provide feedback on generated hypotheses to guide improvements")
        print("  4. Add research focus areas to direct exploration")
        print("  5. Add resources like papers or URLs for deeper knowledge")
        print("  6. Use search and synthesize for targeted knowledge acquisition")
        print("  7. Add your own hypotheses to combine with system-generated ones")
    
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
    
    async def _generate_hypothesis(self) -> int:
        """Generate a new hypothesis.
        
        Returns:
            int: The number of hypotheses generated
        """
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
                
            logger.info(f"Generated {len(hypotheses)} hypotheses")
            return len(hypotheses)
            
        except Exception as e:
            logger.error(f"Error generating hypothesis: {e}", exc_info=True)
            return 0
    
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
    
    async def _evolve_hypothesis(self) -> int:
        """
        Evolve a hypothesis to improve it.
        
        Returns:
            int: The number of hypotheses evolved
        """
        try:
            # Get all hypotheses
            all_hypotheses = self.db.get_hypotheses_for_goal(self.current_research_goal.id)
            
            if not all_hypotheses:
                logger.info("No hypotheses available for evolution")
                return 0
                
            # First, check if there are any hypotheses that haven't been evolved yet
            # These are hypotheses without any parent_hypotheses entries and are not from 'evolved' source
            never_evolved = [h for h in all_hypotheses if not h.parent_hypotheses and h.source != HypothesisSource.EVOLVED]
            
            # If we have never-evolved hypotheses, prioritize them
            if never_evolved:
                logger.info(f"Found {len(never_evolved)} hypotheses that have never been evolved")
                target_hypotheses = never_evolved
            else:
                # Otherwise use all hypotheses
                target_hypotheses = all_hypotheses
            
            # Ensure we have at least one hypothesis to evolve, or we may end up with nothing
            if not target_hypotheses:
                logger.info("No suitable hypotheses found for evolution")
                return 0
            
            # Get the distribution of evolution strategies based on system capabilities
            strategy_weights = {
                "improve": 0.3,
                "combine": 0.2,
                "out_of_box": 0.1,
                "simplify": 0.1,
            }
            
            # If we have domain knowledge available, add those strategies with higher weights
            if hasattr(self, 'domain_knowledge') and self.domain_knowledge.initialized:
                strategy_weights["domain_knowledge"] = 0.2
                strategy_weights["cross_domain"] = 0.2
                # Adjust other weights to maintain sum = 1.0
                total = sum(strategy_weights.values())
                strategy_weights = {k: v/total for k, v in strategy_weights.items()}
            
            # Choose evolution method based on weighted probabilities
            strategies = list(strategy_weights.keys())
            weights = list(strategy_weights.values())
            selected_strategy = random.choices(strategies, weights=weights, k=1)[0]
            
            logger.info(f"Selected evolution strategy: {selected_strategy}")
            
            # Track number of hypotheses evolved
            num_evolved = 0
            
            if selected_strategy == "improve":
                # Improve a single hypothesis
                # Select a hypothesis with preference for higher ratings
                sorted_hypotheses = sorted(target_hypotheses, key=lambda h: h.elo_rating, reverse=True)
                target_idx = int(random.triangular(0, len(sorted_hypotheses) - 1, 0))
                hypothesis = sorted_hypotheses[target_idx]
                
                # Get reviews for this hypothesis
                reviews = self.db.get_reviews_for_hypothesis(hypothesis.id)
                
                # Use standard improvement strategy
                improved = await self.evolution.improve_hypothesis(
                    hypothesis,
                    self.current_research_goal,
                    reviews
                )
                
                # Save the improved hypothesis
                self.db.hypotheses.save(improved)
                
                logger.info(f"Improved hypothesis {hypothesis.id} -> {improved.id}")
                num_evolved += 1
                
            elif selected_strategy == "domain_knowledge":
                # Improve using domain-specific knowledge
                # Select a hypothesis with preference for higher ratings
                sorted_hypotheses = sorted(target_hypotheses, key=lambda h: h.elo_rating, reverse=True)
                target_idx = int(random.triangular(0, len(sorted_hypotheses) - 1, 0))
                hypothesis = sorted_hypotheses[target_idx]
                
                # Get reviews for this hypothesis
                reviews = self.db.get_reviews_for_hypothesis(hypothesis.id)
                
                # Use domain knowledge enhancement
                improved = await self.evolution.improve_with_domain_knowledge(
                    hypothesis,
                    self.current_research_goal,
                    reviews
                )
                
                # Save the improved hypothesis
                self.db.hypotheses.save(improved)
                
                logger.info(f"Improved hypothesis with domain knowledge {hypothesis.id} -> {improved.id}")
                num_evolved += 1
                
            elif selected_strategy == "cross_domain":
                # Apply cross-domain inspiration
                # Select a hypothesis with preference for higher ratings
                sorted_hypotheses = sorted(target_hypotheses, key=lambda h: h.elo_rating, reverse=True)
                target_idx = int(random.triangular(0, len(sorted_hypotheses) - 1, 0))
                hypothesis = sorted_hypotheses[target_idx]
                
                # Get reviews for this hypothesis
                reviews = self.db.get_reviews_for_hypothesis(hypothesis.id)
                
                # Use cross-domain inspiration
                improved = await self.evolution.apply_cross_domain_inspiration(
                    hypothesis,
                    self.current_research_goal,
                    reviews
                )
                
                # Save the improved hypothesis
                self.db.hypotheses.save(improved)
                
                logger.info(f"Applied cross-domain inspiration to hypothesis {hypothesis.id} -> {improved.id}")
                num_evolved += 1
                
            elif selected_strategy == "combine":
                # Combine multiple hypotheses
                # Select 2-3 hypotheses, with preference for higher ratings
                sorted_hypotheses = sorted(target_hypotheses, key=lambda h: h.elo_rating, reverse=True)
                num_to_combine = min(2 + int(random.random() * 2), len(sorted_hypotheses))
                
                if num_to_combine < 2:
                    logger.info("Not enough hypotheses for combination")
                    return
                
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
                num_evolved += 1
                
            elif selected_strategy == "out_of_box":
                # Generate an out-of-box hypothesis
                out_of_box = await self.evolution.generate_out_of_box_hypothesis(
                    self.current_research_goal,
                    existing_hypotheses=all_hypotheses
                )
                
                # Save the out-of-box hypothesis
                self.db.hypotheses.save(out_of_box)
                
                logger.info(f"Generated out-of-box hypothesis -> {out_of_box.id}")
                num_evolved += 1
                
            elif selected_strategy == "simplify":
                # Simplify a hypothesis
                # Select a hypothesis with preference for higher ratings
                sorted_hypotheses = sorted(target_hypotheses, key=lambda h: h.elo_rating, reverse=True)
                target_idx = int(random.triangular(0, len(sorted_hypotheses) - 1, 0))
                hypothesis = sorted_hypotheses[target_idx]
                
                # Simplify the hypothesis
                simplified = await self.evolution.simplify_hypothesis(
                    hypothesis,
                    self.current_research_goal
                )
                
                # Save the simplified hypothesis
                self.db.hypotheses.save(simplified)
                
                logger.info(f"Simplified hypothesis {hypothesis.id} -> {simplified.id}")
                num_evolved += 1
                
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
                    num_evolved += 1
                else:
                    logger.info("No complex hypotheses found for simplification")
            
            return num_evolved
            
        except Exception as e:
            logger.error(f"Error evolving hypothesis: {e}", exc_info=True)
            return 0
    
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
            
            # Use the dedicated protocol generation method
            protocol = await self._generate_protocol_for_hypothesis(hypothesis)
            
            if protocol:
                logger.info(f"Generated experimental protocol {protocol.id} for hypothesis {hypothesis.id}")
            else:
                logger.warning(f"Failed to generate protocol for hypothesis {hypothesis.id}")
                
        except Exception as e:
            logger.error(f"Error generating experimental protocol: {e}", exc_info=True)
            
    async def _generate_protocol_for_hypothesis(self, hypothesis: Hypothesis) -> Optional[ExperimentalProtocol]:
        """
        Generate an experimental protocol for a specific hypothesis.
        
        Args:
            hypothesis (Hypothesis): The hypothesis to generate a protocol for.
            
        Returns:
            Optional[ExperimentalProtocol]: The generated protocol, or None if generation failed.
        """
        try:
            # Check if we already have a protocol for this hypothesis
            existing_protocols = [p for p in self.db.experimental_protocols.get_all() 
                                if p.hypothesis_id == hypothesis.id]
            
            if existing_protocols:
                logger.info(f"Protocol already exists for hypothesis {hypothesis.id}")
                return existing_protocols[0]
            
            # Generate protocol
            protocol = await self.generation.generate_experimental_protocol(
                hypothesis, 
                self.current_research_goal
            )
            
            if protocol:
                # Save protocol to database
                self.db.experimental_protocols.save(protocol)
                
                # Update state counter
                self.current_state["num_protocols"] = self.current_state.get("num_protocols", 0) + 1
                
                return protocol
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error generating protocol for hypothesis {hypothesis.id}: {e}", exc_info=True)
            return None
            
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