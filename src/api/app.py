"""
FastAPI application for Raul Co-Scientist API.
"""

import os
import json
import asyncio
from typing import Dict, List, Any, Optional, Set
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Header, Query, Path
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from ..core.system import CoScientistSystem
from ..core.models import (
    ResearchGoal, 
    Hypothesis, 
    Review, 
    ResearchOverview, 
    UserFeedback, 
    ResearchFocus,
    ReviewType,
    HypothesisSource,
    UserSession
)
from ..config.config import load_config

# Initialize the FastAPI app
app = FastAPI(
    title="Raul Co-Scientist API",
    description="API for the Raul Co-Scientist system",
    version="0.1.0"
)

# System instance will be created when needed
system = None

# Simple API key authentication
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# ---- Request/Response Models ----

class ResearchGoalRequest(BaseModel):
    """Request model for submitting a research goal."""
    text: str
    config_name: Optional[str] = "default"
    user_id: str = Field(..., description="ID of the user submitting the research goal")

class ResearchGoalResponse(BaseModel):
    """Response model for a research goal."""
    id: str
    text: str
    created_at: str
    plan_config: Dict[str, Any]

class HypothesisBase(BaseModel):
    """Base model for hypothesis requests."""
    title: str
    description: str
    summary: str
    supporting_evidence: Optional[List[str]] = None
    tags: Optional[Set[str]] = None

class UserHypothesisRequest(HypothesisBase):
    """Request model for user-submitted hypothesis."""
    user_id: str = Field(..., description="ID of the user submitting the hypothesis")
    research_goal_id: str = Field(..., description="ID of the research goal")
    parent_hypotheses: Optional[List[str]] = None

class UserReviewRequest(BaseModel):
    """Request model for user-submitted review."""
    hypothesis_id: str = Field(..., description="ID of the hypothesis being reviewed")
    text: str = Field(..., description="Full text of the review")
    user_id: str = Field(..., description="ID of the user submitting the review")
    novelty_score: Optional[float] = Field(None, description="Novelty score (0-10)")
    correctness_score: Optional[float] = Field(None, description="Correctness score (0-10)")
    testability_score: Optional[float] = Field(None, description="Testability score (0-10)")
    overall_score: Optional[float] = Field(None, description="Overall score (0-10)")
    critiques: Optional[List[str]] = Field(None, description="Specific critiques")
    strengths: Optional[List[str]] = Field(None, description="Specific strengths")
    improvement_suggestions: Optional[List[str]] = Field(None, description="Suggestions for improvement")

class UserFeedbackRequest(BaseModel):
    """Request model for user feedback."""
    research_goal_id: str = Field(..., description="ID of the research goal")
    feedback_type: str = Field(..., description="Type of feedback (e.g., 'direction', 'focus', 'general')")
    text: str = Field(..., description="Full text of the feedback")
    user_id: str = Field(..., description="ID of the user providing feedback")
    metadata: Optional[Dict[str, Any]] = None

class ResearchFocusRequest(BaseModel):
    """Request model for research focus areas."""
    research_goal_id: str = Field(..., description="ID of the research goal")
    title: str = Field(..., description="Title of the focus area")
    description: str = Field(..., description="Description of the focus area")
    keywords: Optional[List[str]] = None
    priority: float = Field(1.0, description="Priority level (0.0-1.0)")
    user_id: str = Field(..., description="ID of the user specifying this focus")

class UpdateResearchFocusRequest(BaseModel):
    """Request model for updating research focus areas."""
    active: bool = Field(..., description="Whether this focus should be active")
    priority: Optional[float] = Field(None, description="Updated priority level (0.0-1.0)")
    description: Optional[str] = Field(None, description="Updated description")
    keywords: Optional[List[str]] = Field(None, description="Updated keywords")

class TournamentJudgmentRequest(BaseModel):
    """Request model for user judgment in tournament."""
    hypothesis1_id: str = Field(..., description="ID of the first hypothesis")
    hypothesis2_id: str = Field(..., description="ID of the second hypothesis")
    winner_id: Optional[str] = Field(None, description="ID of the winning hypothesis (None for a draw)")
    rationale: str = Field(..., description="Rationale for the decision")
    user_id: str = Field(..., description="ID of the user judging the match")

class UserSessionRequest(BaseModel):
    """Request model for starting a user session."""
    user_id: str = Field(..., description="ID of the user")
    research_goal_id: Optional[str] = Field(None, description="ID of the research goal being worked on")

class HypothesisResponse(BaseModel):
    """Response model for a hypothesis."""
    id: str
    title: str
    summary: str
    description: str
    supporting_evidence: List[str]
    creator: str
    status: str
    source: str
    elo_rating: float
    created_at: str
    user_id: Optional[str] = None
    matches_played: int
    novelty_score: Optional[float] = None
    correctness_score: Optional[float] = None
    testability_score: Optional[float] = None
    tags: Set[str]

class ReviewResponse(BaseModel):
    """Response model for a review."""
    id: str
    hypothesis_id: str
    review_type: str
    reviewer: str
    text: str
    novelty_score: Optional[float]
    correctness_score: Optional[float]
    testability_score: Optional[float]
    overall_score: Optional[float]
    created_at: str
    user_id: Optional[str] = None

class UserFeedbackResponse(BaseModel):
    """Response model for user feedback."""
    id: str
    research_goal_id: str
    feedback_type: str
    text: str
    created_at: str
    user_id: str

class ResearchFocusResponse(BaseModel):
    """Response model for research focus."""
    id: str
    research_goal_id: str
    title: str
    description: str
    keywords: List[str]
    priority: float
    created_at: str
    user_id: str
    active: bool

class TournamentMatchResponse(BaseModel):
    """Response model for tournament match."""
    id: str
    hypothesis1_id: str
    hypothesis2_id: str
    winner_id: Optional[str]
    rationale: str
    judge: str
    created_at: str
    user_id: Optional[str] = None

class SystemStateResponse(BaseModel):
    """Response model for system state."""
    research_goal_id: str
    iterations_completed: int
    num_hypotheses: int
    num_reviews: int
    num_tournament_matches: int
    top_hypotheses: List[HypothesisResponse]
    agent_weights: Dict[str, float]
    active_focus_areas: List[ResearchFocusResponse]
    recent_user_feedback: List[UserFeedbackResponse]

class ResearchOverviewResponse(BaseModel):
    """Response model for research overview."""
    id: str
    title: str
    summary: str
    research_areas: List[Dict[str, Any]]
    top_hypotheses: List[HypothesisResponse]
    created_at: str

class RunIterationRequest(BaseModel):
    """Request model for running iterations."""
    iterations: int = 1

class ErrorResponse(BaseModel):
    """Response model for errors."""
    detail: str

# ---- Background Tasks ----

async def initialize_system(config_name: str = "default"):
    """Initialize the system if not already initialized."""
    global system
    if system is None:
        system = CoScientistSystem(config_name=config_name)
    return system

async def run_iterations_task(iterations: int = 1):
    """Run iterations in the background."""
    global system
    if system is None or system.current_research_goal is None:
        raise ValueError("System not initialized or no research goal set")
        
    for i in range(iterations):
        await system.run_iteration()

# ---- API Routes ----

@app.get("/", tags=["Info"])
async def root():
    """Root endpoint that returns basic information about the API."""
    return {
        "name": "Raul Co-Scientist API",
        "version": "0.1.0",
        "description": "API for the Raul Co-Scientist system"
    }

@app.post("/research_goal", response_model=ResearchGoalResponse, tags=["Research"])
async def create_research_goal(goal: ResearchGoalRequest):
    """Submit a research goal to analyze."""
    global system
    
    # Initialize system with specified config
    system = await initialize_system(goal.config_name)
    
    # Create a user session
    session = UserSession(
        user_id=goal.user_id,
        research_goal_id=None,  # Will be updated after goal creation
        active=True
    )
    
    # Analyze the research goal
    research_goal = await system.analyze_research_goal(goal.text)
    
    # Update and save the user session
    session.research_goal_id = research_goal.id
    system.db.user_sessions.save(session)
    
    # Convert to response model
    return ResearchGoalResponse(
        id=research_goal.id,
        text=research_goal.text,
        created_at=research_goal.created_at.isoformat(),
        plan_config=system.research_plan_config
    )

@app.get("/state", response_model=SystemStateResponse, tags=["System"])
async def get_system_state():
    """Get the current state of the system."""
    global system
    
    if system is None or system.current_research_goal is None:
        raise HTTPException(status_code=404, detail="System not initialized or no research goal set")
    
    # Get top hypotheses
    top_hypotheses = []
    for h_id in system.current_state.get("top_hypotheses", []):
        hypothesis = system.db.hypotheses.get(h_id)
        if hypothesis:
            top_hypotheses.append(HypothesisResponse(
                id=hypothesis.id,
                title=hypothesis.title,
                summary=hypothesis.summary,
                description=hypothesis.description,
                supporting_evidence=hypothesis.supporting_evidence,
                creator=hypothesis.creator,
                status=hypothesis.status,
                source=hypothesis.source,
                elo_rating=hypothesis.elo_rating,
                created_at=hypothesis.created_at.isoformat(),
                user_id=hypothesis.user_id,
                matches_played=hypothesis.matches_played,
                novelty_score=hypothesis.novelty_score,
                correctness_score=hypothesis.correctness_score,
                testability_score=hypothesis.testability_score,
                tags=hypothesis.tags
            ))
    
    # Get active research focus areas
    active_focus_areas = system.db.get_active_research_focus(system.current_research_goal.id)
    focus_responses = [
        ResearchFocusResponse(
            id=focus.id,
            research_goal_id=focus.research_goal_id,
            title=focus.title,
            description=focus.description,
            keywords=focus.keywords,
            priority=focus.priority,
            created_at=focus.created_at.isoformat(),
            user_id=focus.user_id,
            active=focus.active
        )
        for focus in active_focus_areas
    ]
    
    # Get recent user feedback
    recent_feedback = system.db.get_user_feedback(system.current_research_goal.id, limit=5)
    feedback_responses = [
        UserFeedbackResponse(
            id=feedback.id,
            research_goal_id=feedback.research_goal_id,
            feedback_type=feedback.feedback_type,
            text=feedback.text,
            created_at=feedback.created_at.isoformat(),
            user_id=feedback.user_id
        )
        for feedback in recent_feedback
    ]
    
    # Create response
    return SystemStateResponse(
        research_goal_id=system.current_research_goal.id,
        iterations_completed=system.current_state["iterations_completed"],
        num_hypotheses=system.current_state["num_hypotheses"],
        num_reviews=system.current_state["num_reviews"],
        num_tournament_matches=system.current_state["num_tournament_matches"],
        top_hypotheses=top_hypotheses,
        agent_weights=system.agent_weights,
        active_focus_areas=focus_responses,
        recent_user_feedback=feedback_responses
    )

@app.post("/run", tags=["System"])
async def run_iterations(request: RunIterationRequest, background_tasks: BackgroundTasks):
    """Run system iterations in the background."""
    global system
    
    if system is None or system.current_research_goal is None:
        raise HTTPException(status_code=404, detail="System not initialized or no research goal set")
    
    # Run iterations in the background
    background_tasks.add_task(run_iterations_task, request.iterations)
    
    return {
        "message": f"Running {request.iterations} iteration(s) in the background",
        "status": "processing"
    }

@app.get("/hypotheses", response_model=List[HypothesisResponse], tags=["Research"])
async def get_hypotheses(
    limit: int = 10, 
    offset: int = 0,
    user_id: Optional[str] = None, 
    source: Optional[str] = None,
    min_rating: Optional[float] = None
):
    """Get hypotheses with optional filtering."""
    global system
    
    if system is None or system.current_research_goal is None:
        raise HTTPException(status_code=404, detail="System not initialized or no research goal set")
    
    # Get all hypotheses
    all_hypotheses = system.db.hypotheses.get_all()
    
    # Apply filters
    filtered_hypotheses = all_hypotheses
    
    if user_id:
        filtered_hypotheses = [h for h in filtered_hypotheses if h.user_id == user_id]
        
    if source:
        filtered_hypotheses = [h for h in filtered_hypotheses if h.source == source]
        
    if min_rating:
        filtered_hypotheses = [h for h in filtered_hypotheses if h.elo_rating >= min_rating]
    
    # Sort by rating
    sorted_hypotheses = sorted(filtered_hypotheses, key=lambda h: h.elo_rating, reverse=True)
    
    # Apply pagination
    paginated = sorted_hypotheses[offset:offset+limit]
    
    # Convert to response models
    response = [
        HypothesisResponse(
            id=h.id,
            title=h.title,
            summary=h.summary,
            description=h.description,
            supporting_evidence=h.supporting_evidence,
            creator=h.creator,
            status=h.status,
            source=h.source,
            elo_rating=h.elo_rating,
            created_at=h.created_at.isoformat(),
            user_id=h.user_id,
            matches_played=h.matches_played,
            novelty_score=h.novelty_score,
            correctness_score=h.correctness_score,
            testability_score=h.testability_score,
            tags=h.tags
        )
        for h in paginated
    ]
    
    return response

@app.get("/hypotheses/{hypothesis_id}", response_model=HypothesisResponse, tags=["Research"])
async def get_hypothesis(hypothesis_id: str):
    """Get a specific hypothesis by ID."""
    global system
    
    if system is None or system.current_research_goal is None:
        raise HTTPException(status_code=404, detail="System not initialized or no research goal set")
    
    hypothesis = system.db.hypotheses.get(hypothesis_id)
    if not hypothesis:
        raise HTTPException(status_code=404, detail=f"Hypothesis with ID {hypothesis_id} not found")
    
    return HypothesisResponse(
        id=hypothesis.id,
        title=hypothesis.title,
        summary=hypothesis.summary,
        description=hypothesis.description,
        supporting_evidence=hypothesis.supporting_evidence,
        creator=hypothesis.creator,
        status=hypothesis.status,
        source=hypothesis.source,
        elo_rating=hypothesis.elo_rating,
        created_at=hypothesis.created_at.isoformat(),
        user_id=hypothesis.user_id,
        matches_played=hypothesis.matches_played,
        novelty_score=hypothesis.novelty_score,
        correctness_score=hypothesis.correctness_score,
        testability_score=hypothesis.testability_score,
        tags=hypothesis.tags
    )

@app.post("/hypotheses", response_model=HypothesisResponse, tags=["User Interaction"])
async def submit_user_hypothesis(request: UserHypothesisRequest):
    """Submit a user hypothesis."""
    global system
    
    if system is None or system.current_research_goal is None:
        raise HTTPException(status_code=404, detail="System not initialized or no research goal set")
    
    # Check if the research goal exists
    research_goal = system.db.research_goals.get(request.research_goal_id)
    if not research_goal:
        raise HTTPException(status_code=404, detail=f"Research goal with ID {request.research_goal_id} not found")
    
    # Create the hypothesis
    hypothesis = Hypothesis(
        title=request.title,
        description=request.description,
        summary=request.summary,
        supporting_evidence=request.supporting_evidence or [],
        creator="user",
        source=HypothesisSource.USER,
        user_id=request.user_id,
        parent_hypotheses=request.parent_hypotheses or [],
        tags=request.tags or set(),
        metadata={"research_goal_id": request.research_goal_id}
    )
    
    # Save to database
    system.db.hypotheses.save(hypothesis)
    
    # Update statistics
    system.current_state["num_hypotheses"] = len(system.db.hypotheses.get_all())
    
    return HypothesisResponse(
        id=hypothesis.id,
        title=hypothesis.title,
        summary=hypothesis.summary,
        description=hypothesis.description,
        supporting_evidence=hypothesis.supporting_evidence,
        creator=hypothesis.creator,
        status=hypothesis.status,
        source=hypothesis.source,
        elo_rating=hypothesis.elo_rating,
        created_at=hypothesis.created_at.isoformat(),
        user_id=hypothesis.user_id,
        matches_played=hypothesis.matches_played,
        novelty_score=hypothesis.novelty_score,
        correctness_score=hypothesis.correctness_score,
        testability_score=hypothesis.testability_score,
        tags=hypothesis.tags
    )

@app.get("/reviews", response_model=List[ReviewResponse], tags=["Research"])
async def get_reviews(
    hypothesis_id: Optional[str] = None,
    user_id: Optional[str] = None,
    review_type: Optional[str] = None,
    limit: int = 10,
    offset: int = 0
):
    """Get reviews with optional filtering."""
    global system
    
    if system is None or system.current_research_goal is None:
        raise HTTPException(status_code=404, detail="System not initialized or no research goal set")
    
    # Get all reviews
    all_reviews = system.db.reviews.get_all()
    
    # Apply filters
    filtered_reviews = all_reviews
    
    if hypothesis_id:
        filtered_reviews = [r for r in filtered_reviews if r.hypothesis_id == hypothesis_id]
        
    if user_id:
        filtered_reviews = [r for r in filtered_reviews if r.user_id == user_id]
        
    if review_type:
        filtered_reviews = [r for r in filtered_reviews if r.review_type == review_type]
    
    # Sort by creation time (newest first)
    sorted_reviews = sorted(filtered_reviews, key=lambda r: r.created_at, reverse=True)
    
    # Apply pagination
    paginated = sorted_reviews[offset:offset+limit]
    
    # Convert to response models
    response = [
        ReviewResponse(
            id=r.id,
            hypothesis_id=r.hypothesis_id,
            review_type=r.review_type,
            reviewer=r.reviewer,
            text=r.text,
            novelty_score=r.novelty_score,
            correctness_score=r.correctness_score,
            testability_score=r.testability_score,
            overall_score=r.overall_score,
            created_at=r.created_at.isoformat(),
            user_id=r.user_id
        )
        for r in paginated
    ]
    
    return response

@app.post("/reviews", response_model=ReviewResponse, tags=["User Interaction"])
async def submit_user_review(request: UserReviewRequest):
    """Submit a user review for a hypothesis."""
    global system
    
    if system is None or system.current_research_goal is None:
        raise HTTPException(status_code=404, detail="System not initialized or no research goal set")
    
    # Check if the hypothesis exists
    hypothesis = system.db.hypotheses.get(request.hypothesis_id)
    if not hypothesis:
        raise HTTPException(status_code=404, detail=f"Hypothesis with ID {request.hypothesis_id} not found")
    
    # Create the review
    review = Review(
        hypothesis_id=request.hypothesis_id,
        review_type=ReviewType.USER,
        reviewer="user",
        text=request.text,
        novelty_score=request.novelty_score,
        correctness_score=request.correctness_score,
        testability_score=request.testability_score,
        overall_score=request.overall_score,
        critiques=request.critiques or [],
        strengths=request.strengths or [],
        improvement_suggestions=request.improvement_suggestions or [],
        user_id=request.user_id
    )
    
    # Save to database
    system.db.reviews.save(review)
    
    # Update statistics
    system.current_state["num_reviews"] = len(system.db.reviews.get_all())
    
    # Update hypothesis status and scores
    hypothesis.status = "reviewed"
    
    # Update hypothesis scores based on review
    if review.novelty_score is not None:
        hypothesis.novelty_score = review.novelty_score
    if review.correctness_score is not None:
        hypothesis.correctness_score = review.correctness_score
    if review.testability_score is not None:
        hypothesis.testability_score = review.testability_score
    
    # Save updated hypothesis
    system.db.hypotheses.save(hypothesis)
    
    return ReviewResponse(
        id=review.id,
        hypothesis_id=review.hypothesis_id,
        review_type=review.review_type,
        reviewer=review.reviewer,
        text=review.text,
        novelty_score=review.novelty_score,
        correctness_score=review.correctness_score,
        testability_score=review.testability_score,
        overall_score=review.overall_score,
        created_at=review.created_at.isoformat(),
        user_id=review.user_id
    )

@app.post("/feedback", response_model=UserFeedbackResponse, tags=["User Interaction"])
async def submit_user_feedback(request: UserFeedbackRequest):
    """Submit general feedback on the research process."""
    global system
    
    if system is None or system.current_research_goal is None:
        raise HTTPException(status_code=404, detail="System not initialized or no research goal set")
    
    # Check if the research goal exists
    research_goal = system.db.research_goals.get(request.research_goal_id)
    if not research_goal:
        raise HTTPException(status_code=404, detail=f"Research goal with ID {request.research_goal_id} not found")
    
    # Create the feedback
    feedback = UserFeedback(
        research_goal_id=request.research_goal_id,
        feedback_type=request.feedback_type,
        text=request.text,
        user_id=request.user_id,
        metadata=request.metadata or {}
    )
    
    # Save to database
    system.db.user_feedback.save(feedback)
    
    return UserFeedbackResponse(
        id=feedback.id,
        research_goal_id=feedback.research_goal_id,
        feedback_type=feedback.feedback_type,
        text=feedback.text,
        created_at=feedback.created_at.isoformat(),
        user_id=feedback.user_id
    )

@app.post("/focus", response_model=ResearchFocusResponse, tags=["User Interaction"])
async def create_research_focus(request: ResearchFocusRequest):
    """Create a research focus area to guide the exploration."""
    global system
    
    if system is None or system.current_research_goal is None:
        raise HTTPException(status_code=404, detail="System not initialized or no research goal set")
    
    # Check if the research goal exists
    research_goal = system.db.research_goals.get(request.research_goal_id)
    if not research_goal:
        raise HTTPException(status_code=404, detail=f"Research goal with ID {request.research_goal_id} not found")
    
    # Create the research focus
    focus = ResearchFocus(
        research_goal_id=request.research_goal_id,
        title=request.title,
        description=request.description,
        keywords=request.keywords or [],
        priority=request.priority,
        user_id=request.user_id,
        active=True
    )
    
    # Save to database
    system.db.research_focus.save(focus)
    
    return ResearchFocusResponse(
        id=focus.id,
        research_goal_id=focus.research_goal_id,
        title=focus.title,
        description=focus.description,
        keywords=focus.keywords,
        priority=focus.priority,
        created_at=focus.created_at.isoformat(),
        user_id=focus.user_id,
        active=focus.active
    )

@app.put("/focus/{focus_id}", response_model=ResearchFocusResponse, tags=["User Interaction"])
async def update_research_focus(focus_id: str, request: UpdateResearchFocusRequest):
    """Update an existing research focus area."""
    global system
    
    if system is None or system.current_research_goal is None:
        raise HTTPException(status_code=404, detail="System not initialized or no research goal set")
    
    # Get the focus area
    focus = system.db.research_focus.get(focus_id)
    if not focus:
        raise HTTPException(status_code=404, detail=f"Research focus with ID {focus_id} not found")
    
    # Update fields
    focus.active = request.active
    
    if request.priority is not None:
        focus.priority = request.priority
        
    if request.description is not None:
        focus.description = request.description
        
    if request.keywords is not None:
        focus.keywords = request.keywords
    
    # Save to database
    system.db.research_focus.save(focus)
    
    return ResearchFocusResponse(
        id=focus.id,
        research_goal_id=focus.research_goal_id,
        title=focus.title,
        description=focus.description,
        keywords=focus.keywords,
        priority=focus.priority,
        created_at=focus.created_at.isoformat(),
        user_id=focus.user_id,
        active=focus.active
    )

@app.post("/tournament/judge", response_model=TournamentMatchResponse, tags=["User Interaction"])
async def judge_tournament_match(request: TournamentJudgmentRequest):
    """Submit a user judgment for a tournament match between two hypotheses."""
    global system
    
    if system is None or system.current_research_goal is None:
        raise HTTPException(status_code=404, detail="System not initialized or no research goal set")
    
    # Check if both hypotheses exist
    hypothesis1 = system.db.hypotheses.get(request.hypothesis1_id)
    hypothesis2 = system.db.hypotheses.get(request.hypothesis2_id)
    
    if not hypothesis1:
        raise HTTPException(status_code=404, detail=f"Hypothesis with ID {request.hypothesis1_id} not found")
        
    if not hypothesis2:
        raise HTTPException(status_code=404, detail=f"Hypothesis with ID {request.hypothesis2_id} not found")
    
    # Ensure winner ID is valid
    if request.winner_id and request.winner_id not in [request.hypothesis1_id, request.hypothesis2_id]:
        raise HTTPException(status_code=400, detail="Winner ID must be one of the two hypothesis IDs or null for a draw")
    
    # Create the tournament match
    match = TournamentMatch(
        hypothesis1_id=request.hypothesis1_id,
        hypothesis2_id=request.hypothesis2_id,
        winner_id=request.winner_id,
        rationale=request.rationale,
        debate_transcript="User judgment",
        judge="user",
        user_id=request.user_id
    )
    
    # Save to database
    system.db.tournament_matches.save(match)
    
    # Update Elo ratings based on the result
    if request.winner_id:
        # Get current ratings
        rating1 = hypothesis1.elo_rating
        rating2 = hypothesis2.elo_rating
        
        # Calculate expected scores
        expected1 = 1 / (1 + 10 ** ((rating2 - rating1) / 400))
        expected2 = 1 / (1 + 10 ** ((rating1 - rating2) / 400))
        
        # Calculate actual scores
        actual1 = 1.0 if request.winner_id == request.hypothesis1_id else 0.0
        actual2 = 1.0 if request.winner_id == request.hypothesis2_id else 0.0
        
        # Update ratings (K-factor of 32)
        hypothesis1.elo_rating = rating1 + 32 * (actual1 - expected1)
        hypothesis2.elo_rating = rating2 + 32 * (actual2 - expected2)
        
        # Update matches played
        hypothesis1.matches_played += 1
        hypothesis2.matches_played += 1
        
        # Save updated hypotheses
        system.db.hypotheses.save(hypothesis1)
        system.db.hypotheses.save(hypothesis2)
    
    # Update statistics
    system.current_state["num_tournament_matches"] = len(system.db.tournament_matches.get_all())
    
    return TournamentMatchResponse(
        id=match.id,
        hypothesis1_id=match.hypothesis1_id,
        hypothesis2_id=match.hypothesis2_id,
        winner_id=match.winner_id,
        rationale=match.rationale,
        judge=match.judge,
        created_at=match.created_at.isoformat(),
        user_id=match.user_id
    )

@app.get("/tournament/pair", tags=["User Interaction"])
async def get_tournament_pair():
    """Get a pair of hypotheses for the user to judge."""
    global system
    
    if system is None or system.current_research_goal is None:
        raise HTTPException(status_code=404, detail="System not initialized or no research goal set")
    
    # Get all hypotheses that have been reviewed
    all_hypotheses = system.db.hypotheses.get_all()
    reviewed_hypotheses = [h for h in all_hypotheses if h.status == "reviewed"]
    
    if len(reviewed_hypotheses) < 2:
        raise HTTPException(status_code=400, detail="Not enough reviewed hypotheses for a tournament match")
    
    # Select two hypotheses for comparison
    # Prioritize hypotheses with fewer matches and higher ratings
    sorted_hypotheses = sorted(
        reviewed_hypotheses,
        key=lambda h: (-h.elo_rating, h.matches_played)
    )
    
    # Get the top 5 hypotheses
    top_hypotheses = sorted_hypotheses[:min(5, len(sorted_hypotheses))]
    
    # Select two different hypotheses
    hypothesis1 = top_hypotheses[0]
    hypothesis2 = top_hypotheses[1] if len(top_hypotheses) > 1 else sorted_hypotheses[5] if len(sorted_hypotheses) > 5 else sorted_hypotheses[-1]
    
    return {
        "hypothesis1": HypothesisResponse(
            id=hypothesis1.id,
            title=hypothesis1.title,
            summary=hypothesis1.summary,
            description=hypothesis1.description,
            supporting_evidence=hypothesis1.supporting_evidence,
            creator=hypothesis1.creator,
            status=hypothesis1.status,
            source=hypothesis1.source,
            elo_rating=hypothesis1.elo_rating,
            created_at=hypothesis1.created_at.isoformat(),
            user_id=hypothesis1.user_id,
            matches_played=hypothesis1.matches_played,
            novelty_score=hypothesis1.novelty_score,
            correctness_score=hypothesis1.correctness_score,
            testability_score=hypothesis1.testability_score,
            tags=hypothesis1.tags
        ),
        "hypothesis2": HypothesisResponse(
            id=hypothesis2.id,
            title=hypothesis2.title,
            summary=hypothesis2.summary,
            description=hypothesis2.description,
            supporting_evidence=hypothesis2.supporting_evidence,
            creator=hypothesis2.creator,
            status=hypothesis2.status,
            source=hypothesis2.source,
            elo_rating=hypothesis2.elo_rating,
            created_at=hypothesis2.created_at.isoformat(),
            user_id=hypothesis2.user_id,
            matches_played=hypothesis2.matches_played,
            novelty_score=hypothesis2.novelty_score,
            correctness_score=hypothesis2.correctness_score,
            testability_score=hypothesis2.testability_score,
            tags=hypothesis2.tags
        )
    }

@app.get("/overview", response_model=ResearchOverviewResponse, tags=["Research"])
async def get_research_overview():
    """Get the latest research overview."""
    global system
    
    if system is None or system.current_research_goal is None:
        raise HTTPException(status_code=404, detail="System not initialized or no research goal set")
    
    # Get the latest research overview
    last_overview_id = system.current_state.get("last_research_overview_id")
    
    if last_overview_id:
        overview = system.db.research_overviews.get(last_overview_id)
    else:
        # Generate a new overview if none exists
        overview = await system._generate_research_overview()
        
    if not overview:
        raise HTTPException(status_code=404, detail="No research overview available")
    
    # Get top hypotheses
    top_hypotheses = []
    for h_id in overview.top_hypotheses:
        hypothesis = system.db.hypotheses.get(h_id)
        if hypothesis:
            top_hypotheses.append(HypothesisResponse(
                id=hypothesis.id,
                title=hypothesis.title,
                summary=hypothesis.summary,
                description=hypothesis.description,
                supporting_evidence=hypothesis.supporting_evidence,
                creator=hypothesis.creator,
                status=hypothesis.status,
                source=hypothesis.source,
                elo_rating=hypothesis.elo_rating,
                created_at=hypothesis.created_at.isoformat(),
                user_id=hypothesis.user_id,
                matches_played=hypothesis.matches_played,
                novelty_score=hypothesis.novelty_score,
                correctness_score=hypothesis.correctness_score,
                testability_score=hypothesis.testability_score,
                tags=hypothesis.tags
            ))
    
    # Create response
    return ResearchOverviewResponse(
        id=overview.id,
        title=overview.title,
        summary=overview.summary,
        research_areas=overview.research_areas,
        top_hypotheses=top_hypotheses,
        created_at=overview.created_at.isoformat()
    )

@app.post("/sessions", tags=["User Interaction"])
async def create_user_session(request: UserSessionRequest):
    """Create a new user session."""
    global system
    
    if system is None:
        system = await initialize_system()
    
    # Close any existing active sessions for this user
    active_session = system.db.get_active_user_session(request.user_id)
    if active_session:
        active_session.active = False
        active_session.session_end = datetime.now()
        system.db.user_sessions.save(active_session)
    
    # Create new session
    session = UserSession(
        user_id=request.user_id,
        research_goal_id=request.research_goal_id,
        active=True
    )
    
    # Save to database
    system.db.user_sessions.save(session)
    
    return {
        "session_id": session.id,
        "user_id": session.user_id,
        "research_goal_id": session.research_goal_id,
        "session_start": session.session_start.isoformat(),
        "active": session.active
    }

# Run the API server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)