"""
FastAPI application for Watson Co-Scientist API.
"""

import os
import json
import asyncio
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

from ..core.system import CoScientistSystem
from ..core.models import ResearchGoal, Hypothesis, Review, ResearchOverview
from ..config.config import load_config

# Initialize the FastAPI app
app = FastAPI(
    title="Watson Co-Scientist API",
    description="API for the Watson Co-Scientist system",
    version="0.1.0"
)

# System instance will be created when needed
system = None

# ---- Request/Response Models ----

class ResearchGoalRequest(BaseModel):
    """Request model for submitting a research goal."""
    text: str
    config_name: Optional[str] = "default"

class ResearchGoalResponse(BaseModel):
    """Response model for a research goal."""
    id: str
    text: str
    created_at: str
    plan_config: Dict[str, Any]

class HypothesisResponse(BaseModel):
    """Response model for a hypothesis."""
    id: str
    title: str
    summary: str
    description: str
    supporting_evidence: List[str]
    creator: str
    status: str
    elo_rating: float
    created_at: str

class SystemStateResponse(BaseModel):
    """Response model for system state."""
    research_goal_id: str
    iterations_completed: int
    num_hypotheses: int
    num_reviews: int
    num_tournament_matches: int
    top_hypotheses: List[HypothesisResponse]
    agent_weights: Dict[str, float]

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
        "name": "Watson Co-Scientist API",
        "version": "0.1.0",
        "description": "API for the Watson Co-Scientist system"
    }

@app.post("/research_goal", response_model=ResearchGoalResponse, tags=["Research"])
async def create_research_goal(goal: ResearchGoalRequest):
    """Submit a research goal to analyze."""
    global system
    
    # Initialize system with specified config
    system = await initialize_system(goal.config_name)
    
    # Analyze the research goal
    research_goal = await system.analyze_research_goal(goal.text)
    
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
                elo_rating=hypothesis.elo_rating,
                created_at=hypothesis.created_at.isoformat()
            ))
    
    # Create response
    return SystemStateResponse(
        research_goal_id=system.current_research_goal.id,
        iterations_completed=system.current_state["iterations_completed"],
        num_hypotheses=system.current_state["num_hypotheses"],
        num_reviews=system.current_state["num_reviews"],
        num_tournament_matches=system.current_state["num_tournament_matches"],
        top_hypotheses=top_hypotheses,
        agent_weights=system.agent_weights
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
async def get_hypotheses(limit: int = 10, offset: int = 0):
    """Get all hypotheses."""
    global system
    
    if system is None or system.current_research_goal is None:
        raise HTTPException(status_code=404, detail="System not initialized or no research goal set")
    
    # Get all hypotheses
    all_hypotheses = system.db.hypotheses.get_all()
    sorted_hypotheses = sorted(all_hypotheses, key=lambda h: h.elo_rating, reverse=True)
    
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
            elo_rating=h.elo_rating,
            created_at=h.created_at.isoformat()
        )
        for h in paginated
    ]
    
    return response

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
                elo_rating=hypothesis.elo_rating,
                created_at=hypothesis.created_at.isoformat()
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

# Run the API server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)