"""
Data models for the Co-Scientist system.
"""

import uuid
from enum import Enum
from datetime import datetime
from typing import Dict, List, Optional, Set, Any, Union
from pydantic import BaseModel, Field, validator

class ResearchGoal(BaseModel):
    """A research goal specifies the objective for the co-scientist system."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str = Field(..., description="The text of the research goal")
    created_at: datetime = Field(default_factory=datetime.now)
    preferences: Dict[str, Any] = Field(default_factory=dict, description="User preferences for this research goal")
    constraints: Dict[str, Any] = Field(default_factory=dict, description="Constraints for this research goal")
    
    class Config:
        validate_assignment = True

class HypothesisStatus(str, Enum):
    """Status of a hypothesis."""
    GENERATED = "generated"
    REVIEWED = "reviewed"
    ACCEPTED = "accepted"
    REJECTED = "rejected"

class HypothesisSource(str, Enum):
    """Source of a hypothesis."""
    SYSTEM = "system"
    USER = "user"
    EVOLVED = "evolved"
    COMBINED = "combined"

class Citation(BaseModel):
    """A scientific citation to a research paper or source."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str = Field(..., description="Title of the cited work")
    authors: List[str] = Field(default_factory=list, description="Authors of the cited work")
    year: Optional[int] = Field(None, description="Publication year")
    journal: Optional[str] = Field(None, description="Journal or conference name")
    url: Optional[str] = Field(None, description="URL to the cited work")
    doi: Optional[str] = Field(None, description="Digital Object Identifier")
    publication_date: Optional[str] = Field(None, description="Full publication date if available")
    snippet: Optional[str] = Field(None, description="Relevant snippet or abstract")
    relevance_score: Optional[float] = Field(None, description="Relevance score for this citation")
    created_at: datetime = Field(default_factory=datetime.now)
    source: str = Field(default="search", description="Source of the citation (search, user, etc.)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        validate_assignment = True

class Hypothesis(BaseModel):
    """A scientific hypothesis generated or evaluated by the system."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str = Field(..., description="Title of the hypothesis")
    description: str = Field(..., description="Full description of the hypothesis")
    summary: str = Field(..., description="Brief summary of the hypothesis")
    supporting_evidence: List[str] = Field(default_factory=list, description="Supporting evidence for the hypothesis")
    citations: List[Union[Citation, str]] = Field(default_factory=list, description="Citations supporting this hypothesis")
    creator: str = Field(..., description="Creator of the hypothesis (agent name or 'user')")
    status: HypothesisStatus = Field(default=HypothesisStatus.GENERATED)
    source: HypothesisSource = Field(default=HypothesisSource.SYSTEM, description="Source of the hypothesis")
    created_at: datetime = Field(default_factory=datetime.now)
    parent_hypotheses: List[str] = Field(default_factory=list, description="IDs of parent hypotheses, if this was derived from others")
    elo_rating: float = Field(default=1200, description="Elo rating for tournament ranking")
    matches_played: int = Field(default=0, description="Number of tournament matches played")
    novelty_score: Optional[float] = Field(None, description="Novelty score from reviews")
    correctness_score: Optional[float] = Field(None, description="Correctness score from reviews")
    testability_score: Optional[float] = Field(None, description="Testability score from reviews")
    tags: Set[str] = Field(default_factory=set, description="Tags for this hypothesis")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    user_id: Optional[str] = Field(None, description="ID of the user who created this hypothesis (if user-submitted)")
    literature_grounded: bool = Field(default=False, description="Whether this hypothesis is grounded in literature")
    
    class Config:
        validate_assignment = True
        
    @validator('citations', pre=True)
    def validate_citations(cls, v):
        """Validate that citations are either Citation objects or citation IDs (strings)."""
        if v is None:
            return []
        return v

class ExperimentalProtocol(BaseModel):
    """A protocol for testing a scientific hypothesis."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    hypothesis_id: str = Field(..., description="ID of the hypothesis this protocol tests")
    title: str = Field(..., description="Title of the protocol")
    description: str = Field(..., description="Full description of the protocol")
    steps: List[str] = Field(..., description="Steps to perform the experiment")
    materials: List[str] = Field(default_factory=list, description="Materials needed for the experiment")
    equipment: List[str] = Field(default_factory=list, description="Equipment needed for the experiment")
    expected_results: str = Field(..., description="Expected results from the experiment")
    limitations: List[str] = Field(default_factory=list, description="Limitations of the protocol")
    creator: str = Field(..., description="Creator of the protocol (agent name or 'user')")
    created_at: datetime = Field(default_factory=datetime.now)
    user_id: Optional[str] = Field(None, description="ID of the user who created this protocol (if user-submitted)")
    
    class Config:
        validate_assignment = True

class ReviewType(str, Enum):
    """Type of a review."""
    INITIAL = "initial"
    FULL = "full"
    DEEP_VERIFICATION = "deep_verification"
    OBSERVATION = "observation"
    SIMULATION = "simulation"
    TOURNAMENT = "tournament"
    USER = "user"

class Review(BaseModel):
    """A review of a scientific hypothesis or experimental protocol."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    hypothesis_id: str = Field(..., description="ID of the hypothesis being reviewed")
    review_type: ReviewType = Field(..., description="Type of review")
    reviewer: str = Field(..., description="Name of the reviewer (agent or user)")
    text: str = Field(..., description="Full text of the review")
    novelty_score: Optional[float] = Field(None, description="Novelty score (0-10)")
    correctness_score: Optional[float] = Field(None, description="Correctness score (0-10)")
    testability_score: Optional[float] = Field(None, description="Testability score (0-10)")
    overall_score: Optional[float] = Field(None, description="Overall score (0-10)")
    critiques: List[str] = Field(default_factory=list, description="Specific critiques")
    strengths: List[str] = Field(default_factory=list, description="Specific strengths")
    improvement_suggestions: List[str] = Field(default_factory=list, description="Suggestions for improvement")
    created_at: datetime = Field(default_factory=datetime.now)
    user_id: Optional[str] = Field(None, description="ID of the user who created this review (if user-submitted)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        validate_assignment = True

class UserFeedback(BaseModel):
    """Feedback from a user on the system's output or process."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    research_goal_id: str = Field(..., description="ID of the research goal this feedback relates to")
    hypothesis_id: Optional[str] = Field(None, description="ID of a specific hypothesis this feedback relates to (if applicable)")
    feedback_type: str = Field("general", description="Type of feedback (e.g., 'direction', 'critique', 'improvement', 'resource', 'context', 'general')")
    text: str = Field(..., description="Full text of the feedback")
    suggested_improvements: Optional[str] = Field(None, description="Specific improvements suggested by the user")
    resources: List[Dict[str, str]] = Field(default_factory=list, description="Resources provided by the user (URLs, papers, etc.)")
    priority: int = Field(default=1, description="Priority level (1-5), with 5 being highest priority")
    requires_action: bool = Field(default=False, description="Whether the system needs to take specific action based on this feedback")
    action_taken: bool = Field(default=False, description="Whether action has been taken on this feedback")
    created_at: datetime = Field(default_factory=datetime.now)
    user_id: str = Field(..., description="ID of the user providing feedback")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        validate_assignment = True

class ResearchFocus(BaseModel):
    """A focus area specified by the user to direct the system's exploration."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    research_goal_id: str = Field(..., description="ID of the research goal this focus relates to")
    title: str = Field(..., description="Title of the focus area")
    description: str = Field(..., description="Description of the focus area")
    keywords: List[str] = Field(default_factory=list, description="Keywords related to this focus area")
    priority: float = Field(default=1.0, description="Priority level (0.0-1.0)")
    created_at: datetime = Field(default_factory=datetime.now)
    user_id: str = Field(..., description="ID of the user specifying this focus")
    active: bool = Field(default=True, description="Whether this focus is currently active")
    
    class Config:
        validate_assignment = True

class TournamentMatch(BaseModel):
    """A tournament match between two hypotheses."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    hypothesis1_id: str = Field(..., description="ID of the first hypothesis")
    hypothesis2_id: str = Field(..., description="ID of the second hypothesis")
    winner_id: Optional[str] = Field(None, description="ID of the winning hypothesis (None for a draw)")
    rationale: str = Field(..., description="Rationale for the decision")
    debate_transcript: str = Field(..., description="Transcript of the debate")
    judge: str = Field(..., description="Name of the judge (agent or user)")
    created_at: datetime = Field(default_factory=datetime.now)
    user_id: Optional[str] = Field(None, description="ID of the user who judged this match (if user-judged)")
    
    class Config:
        validate_assignment = True

class ResearchOverview(BaseModel):
    """An overview of research areas and directions related to a research goal."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    research_goal_id: str = Field(..., description="ID of the research goal")
    title: str = Field(..., description="Title of the research overview")
    summary: str = Field(..., description="Summary of the research overview")
    research_areas: List[Dict[str, Any]] = Field(..., description="Research areas identified")
    top_hypotheses: List[str] = Field(default_factory=list, description="IDs of top hypotheses")
    potential_contacts: List[Dict[str, str]] = Field(default_factory=list, description="Potential research contacts")
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        validate_assignment = True

class UserSession(BaseModel):
    """User session information for tracking user interactions with the system."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = Field(..., description="ID of the user")
    session_start: datetime = Field(default_factory=datetime.now)
    session_end: Optional[datetime] = Field(None, description="End time of the session")
    research_goal_id: Optional[str] = Field(None, description="ID of the research goal being worked on")
    active: bool = Field(default=True, description="Whether this session is currently active")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        validate_assignment = True

class MetaReview(BaseModel):
    """A meta-review synthesizing insights from all reviews."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    research_goal_id: str = Field(..., description="ID of the research goal")
    common_issues: List[str] = Field(..., description="Common issues identified across reviews")
    improvement_areas: List[str] = Field(..., description="Areas for improvement")
    successful_approaches: List[str] = Field(..., description="Approaches that were successful")
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        validate_assignment = True