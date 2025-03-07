"""
Database module for storing and retrieving data in the Co-Scientist system.
"""

import json
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, TypeVar, Type, Generic

from pydantic import BaseModel

from .models import (
    ResearchGoal, 
    Hypothesis, 
    ExperimentalProtocol, 
    Review, 
    TournamentMatch, 
    ResearchOverview,
    MetaReview,
    UserFeedback,
    ResearchFocus,
    UserSession,
    Citation
)

logger = logging.getLogger("co_scientist")

T = TypeVar('T', bound=BaseModel)

class JSONRepository(Generic[T]):
    """Repository for storing and retrieving objects as JSON files."""
    
    def __init__(self, data_dir: str, model_class: Type[T]):
        """
        Initialize the repository.
        
        Args:
            data_dir (str): The directory to store the data in.
            model_class (Type[T]): The model class to store.
        """
        self.data_dir = data_dir
        self.model_class = model_class
        self.items: Dict[str, T] = {}
        
        # Create the data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Load all items from the data directory
        self._load_items()
        
    def _load_items(self):
        """Load all items from the data directory."""
        try:
            for filename in os.listdir(self.data_dir):
                if filename.endswith(".json"):
                    filepath = os.path.join(self.data_dir, filename)
                    try:
                        with open(filepath, "r") as f:
                            data = json.load(f)
                            item = self.model_class.parse_obj(data)
                            self.items[item.id] = item
                    except Exception as e:
                        logger.error(f"Error loading {filepath}: {e}")
        except FileNotFoundError:
            # Directory doesn't exist yet, will be created when first item is saved
            logger.warning(f"Directory {self.data_dir} not found, will be created when first item is saved")
    
    def save(self, item: T) -> T:
        """
        Save an item to the repository.
        
        Args:
            item (T): The item to save.
            
        Returns:
            T: The saved item.
        """
        # Add to the in-memory dict
        self.items[item.id] = item
        
        # Save to disk
        filepath = os.path.join(self.data_dir, f"{item.id}.json")
        try:
            with open(filepath, "w") as f:
                json.dump(item.dict(), f, indent=2, default=self._json_serializer)
            logger.debug(f"Saved {self.model_class.__name__} {item.id} to {filepath}")
        except Exception as e:
            logger.error(f"Error saving {filepath}: {e}")
            
        return item
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for objects that are not JSON serializable."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        if hasattr(obj, "__iter__") and not isinstance(obj, (dict, list, str)):
            return list(obj)
        raise TypeError(f"Type {type(obj)} not serializable")
    
    def get(self, id: str) -> Optional[T]:
        """
        Get an item by ID.
        
        Args:
            id (str): The ID of the item to get.
            
        Returns:
            Optional[T]: The item, or None if not found.
        """
        return self.items.get(id)
    
    def get_all(self) -> List[T]:
        """
        Get all items.
        
        Returns:
            List[T]: All items.
        """
        return list(self.items.values())
    
    def find_by(self, field: str, value: Any) -> List[T]:
        """
        Find items where the given field matches the given value.
        
        Args:
            field (str): The field to match
            value (Any): The value to match
            
        Returns:
            List[T]: Matching items
        """
        results = []
        for item in self.items.values():
            item_value = getattr(item, field, None)
            if item_value == value:
                results.append(item)
        return results
    
    def delete(self, id: str) -> bool:
        """
        Delete an item by ID.
        
        Args:
            id (str): The ID of the item to delete.
            
        Returns:
            bool: True if the item was deleted, False otherwise.
        """
        if id in self.items:
            del self.items[id]
            
            # Delete from disk
            filepath = os.path.join(self.data_dir, f"{id}.json")
            if os.path.exists(filepath):
                os.remove(filepath)
                logger.debug(f"Deleted {filepath}")
                return True
        
        return False

class Database:
    """Database for the Co-Scientist system."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the database.
        
        Args:
            data_dir (str): The directory to store the data in.
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Create repositories for each model
        self.research_goals = JSONRepository(
            os.path.join(data_dir, "research_goals"),
            ResearchGoal
        )
        self.hypotheses = JSONRepository(
            os.path.join(data_dir, "hypotheses"),
            Hypothesis
        )
        self.experimental_protocols = JSONRepository(
            os.path.join(data_dir, "experimental_protocols"),
            ExperimentalProtocol
        )
        self.reviews = JSONRepository(
            os.path.join(data_dir, "reviews"),
            Review
        )
        self.tournament_matches = JSONRepository(
            os.path.join(data_dir, "tournament_matches"),
            TournamentMatch
        )
        self.research_overviews = JSONRepository(
            os.path.join(data_dir, "research_overviews"),
            ResearchOverview
        )
        self.meta_reviews = JSONRepository(
            os.path.join(data_dir, "meta_reviews"),
            MetaReview
        )
        
        # New repositories for user interaction
        self.user_feedback = JSONRepository(
            os.path.join(data_dir, "user_feedback"),
            UserFeedback
        )
        self.research_focus = JSONRepository(
            os.path.join(data_dir, "research_focus"),
            ResearchFocus
        )
        self.user_sessions = JSONRepository(
            os.path.join(data_dir, "user_sessions"),
            UserSession
        )
        
        # Add repository for citations
        self.citations = JSONRepository(
            os.path.join(data_dir, "citations"),
            Citation
        )
        
        logger.info(f"Initialized database in {data_dir}")
    
    def get_hypotheses_for_goal(self, research_goal_id: str) -> List[Hypothesis]:
        """
        Get all hypotheses for a research goal.
        
        Args:
            research_goal_id (str): The ID of the research goal.
            
        Returns:
            List[Hypothesis]: The hypotheses.
        """
        # In a real implementation, we would use a database query or an index
        # For now, we'll scan all hypotheses and filter by research goal ID in metadata
        return [
            h for h in self.hypotheses.get_all()
            if h.metadata.get("research_goal_id") == research_goal_id
        ]
    
    def get_user_hypotheses(self, user_id: str, research_goal_id: Optional[str] = None) -> List[Hypothesis]:
        """
        Get all hypotheses submitted by a specific user.
        
        Args:
            user_id (str): The ID of the user.
            research_goal_id (Optional[str]): If provided, filter by research goal ID.
            
        Returns:
            List[Hypothesis]: The user's hypotheses.
        """
        user_hypotheses = [
            h for h in self.hypotheses.get_all()
            if h.user_id == user_id
        ]
        
        if research_goal_id:
            user_hypotheses = [
                h for h in user_hypotheses
                if h.metadata.get("research_goal_id") == research_goal_id
            ]
            
        return user_hypotheses
    
    def get_active_research_focus(self, research_goal_id: str) -> List[ResearchFocus]:
        """
        Get all active research focus areas for a research goal.
        
        Args:
            research_goal_id (str): The ID of the research goal.
            
        Returns:
            List[ResearchFocus]: The active research focus areas.
        """
        return [
            f for f in self.research_focus.get_all()
            if f.research_goal_id == research_goal_id and f.active
        ]
    
    def get_user_feedback(self, research_goal_id: str, limit: int = 10) -> List[UserFeedback]:
        """
        Get recent user feedback for a research goal.
        
        Args:
            research_goal_id (str): The ID of the research goal.
            limit (int): Maximum number of feedback items to return.
            
        Returns:
            List[UserFeedback]: The user feedback.
        """
        feedback = [
            f for f in self.user_feedback.get_all()
            if f.research_goal_id == research_goal_id
        ]
        
        # Sort by creation time (newest first) and apply limit
        feedback.sort(key=lambda f: f.created_at, reverse=True)
        return feedback[:limit]
    
    def get_active_user_session(self, user_id: str) -> Optional[UserSession]:
        """
        Get the active user session for a user.
        
        Args:
            user_id (str): The ID of the user.
            
        Returns:
            Optional[UserSession]: The active user session, or None if not found.
        """
        active_sessions = [
            s for s in self.user_sessions.get_all()
            if s.user_id == user_id and s.active
        ]
        
        if active_sessions:
            return active_sessions[0]
        
        return None
    
    def get_reviews_for_hypothesis(self, hypothesis_id: str) -> List[Review]:
        """
        Get all reviews for a hypothesis.
        
        Args:
            hypothesis_id (str): The ID of the hypothesis.
            
        Returns:
            List[Review]: The reviews.
        """
        return [
            r for r in self.reviews.get_all()
            if r.hypothesis_id == hypothesis_id
        ]
    
    def get_user_reviews(self, user_id: str, research_goal_id: Optional[str] = None) -> List[Review]:
        """
        Get all reviews submitted by a specific user.
        
        Args:
            user_id (str): The ID of the user.
            research_goal_id (Optional[str]): If provided, filter by research goal ID.
            
        Returns:
            List[Review]: The user's reviews.
        """
        user_reviews = [
            r for r in self.reviews.get_all()
            if r.user_id == user_id
        ]
        
        if research_goal_id:
            # Filter by research goal ID using the hypothesis's metadata
            filtered_reviews = []
            for review in user_reviews:
                hypothesis = self.hypotheses.get(review.hypothesis_id)
                if hypothesis and hypothesis.metadata.get("research_goal_id") == research_goal_id:
                    filtered_reviews.append(review)
            return filtered_reviews
            
        return user_reviews
    
    def get_matches_for_hypothesis(self, hypothesis_id: str) -> List[TournamentMatch]:
        """
        Get all tournament matches for a hypothesis.
        
        Args:
            hypothesis_id (str): The ID of the hypothesis.
            
        Returns:
            List[TournamentMatch]: The tournament matches.
        """
        return [
            m for m in self.tournament_matches.get_all()
            if m.hypothesis1_id == hypothesis_id or m.hypothesis2_id == hypothesis_id
        ]
    
    def get_latest_research_overview(self, research_goal_id: str) -> Optional[ResearchOverview]:
        """
        Get the latest research overview for a research goal.
        
        Args:
            research_goal_id (str): The ID of the research goal.
            
        Returns:
            Optional[ResearchOverview]: The latest research overview, or None if not found.
        """
        overviews = [
            o for o in self.research_overviews.get_all()
            if o.research_goal_id == research_goal_id
        ]
        
        if overviews:
            return max(overviews, key=lambda o: o.created_at)
        
        return None
    
    def get_latest_meta_review(self, research_goal_id: str) -> Optional[MetaReview]:
        """
        Get the latest meta-review for a research goal.
        
        Args:
            research_goal_id (str): The ID of the research goal.
            
        Returns:
            Optional[MetaReview]: The latest meta-review, or None if not found.
        """
        meta_reviews = [
            m for m in self.meta_reviews.get_all()
            if m.research_goal_id == research_goal_id
        ]
        
        if meta_reviews:
            return max(meta_reviews, key=lambda m: m.created_at)
        
        return None
        
    def get_citations_for_hypothesis(self, hypothesis_id: str) -> List[Citation]:
        """
        Get all citations for a hypothesis.
        
        Args:
            hypothesis_id (str): The ID of the hypothesis.
            
        Returns:
            List[Citation]: The citations.
        """
        hypothesis = self.hypotheses.get(hypothesis_id)
        if not hypothesis:
            return []
            
        # Get direct citation objects stored in the hypothesis.citations
        direct_citations = []
        for citation in hypothesis.citations:
            if isinstance(citation, str):
                # It's a citation ID, fetch the citation
                citation_obj = self.citations.get(citation)
                if citation_obj:
                    direct_citations.append(citation_obj)
            else:
                # It's already a Citation object
                direct_citations.append(citation)
                
        return direct_citations
        
    def add_citation_to_hypothesis(self, hypothesis_id: str, citation: Citation) -> bool:
        """
        Add a citation to a hypothesis.
        
        Args:
            hypothesis_id (str): The ID of the hypothesis.
            citation (Citation): The citation to add.
            
        Returns:
            bool: True if the citation was added, False otherwise.
        """
        hypothesis = self.hypotheses.get(hypothesis_id)
        if not hypothesis:
            return False
            
        # Save the citation first
        self.citations.save(citation)
        
        # Add the citation ID to the hypothesis
        if citation.id not in [c if isinstance(c, str) else c.id for c in hypothesis.citations]:
            hypothesis.citations.append(citation.id)
            hypothesis.literature_grounded = True
            self.hypotheses.save(hypothesis)
            return True
            
        return False
        
    def get_citations_by_url(self, url: str) -> List[Citation]:
        """
        Get all citations with the given URL.
        
        Args:
            url (str): The URL to match.
            
        Returns:
            List[Citation]: The matching citations.
        """
        return [
            c for c in self.citations.get_all()
            if c.url == url
        ]
        
    def search_citations(self, query: str) -> List[Citation]:
        """
        Search for citations matching the query.
        
        Args:
            query (str): The search query.
            
        Returns:
            List[Citation]: The matching citations.
        """
        # Basic search implementation
        query = query.lower()
        results = []
        
        for citation in self.citations.get_all():
            # Search in title, authors, journal, and snippet
            if (query in citation.title.lower() or
                any(query in author.lower() for author in citation.authors) or
                (citation.journal and query in citation.journal.lower()) or
                (citation.snippet and query in citation.snippet.lower())):
                results.append(citation)
                
        return results