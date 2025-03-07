"""
Meta-Review Agent for synthesizing insights from reviews and tournaments.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Set

from .base_agent import BaseAgent
from ..config.config import SystemConfig
from ..core.models import (
    Hypothesis, 
    ResearchGoal, 
    Review, 
    TournamentMatch, 
    ResearchOverview,
    MetaReview,
    ExperimentalProtocol
)
from ..tools.web_search import WebSearchTool

logger = logging.getLogger("co_scientist")

class MetaReviewAgent(BaseAgent):
    """
    Agent responsible for synthesizing insights from reviews and tournaments.
    """
    
    def __init__(self, config: SystemConfig):
        """
        Initialize the Meta-review agent.
        
        Args:
            config (SystemConfig): The system configuration.
        """
        super().__init__("meta_review", config)
        
        # Initialize web search tool if enabled
        self.web_search = None
        if config.web_search_enabled:
            self.web_search = WebSearchTool(config.web_search_api_key)
    
    async def synthesize_reviews(self, 
                            reviews: List[Review], 
                            research_goal: ResearchGoal) -> MetaReview:
        """
        Synthesize insights from multiple reviews.
        
        Args:
            reviews (List[Review]): The reviews to synthesize.
            research_goal (ResearchGoal): The research goal.
            
        Returns:
            MetaReview: The meta-review.
        """
        logger.info(f"Synthesizing {len(reviews)} reviews for research goal {research_goal.id}")
        
        # Prepare review text
        review_summaries = []
        for i, review in enumerate(reviews, 1):
            summary = f"Review {i} (ID: {review.id}):\n"
            summary += f"Type: {review.review_type}\n"
            summary += f"Scores: Novelty={review.novelty_score}, Correctness={review.correctness_score}, Testability={review.testability_score}, Overall={review.overall_score}\n"
            summary += f"Critiques: {', '.join(review.critiques)}\n"
            summary += f"Strengths: {', '.join(review.strengths)}\n"
            if review.improvement_suggestions:
                summary += f"Improvement Suggestions: {', '.join(review.improvement_suggestions)}\n"
            review_summaries.append(summary)
        
        review_text = "\n\n".join(review_summaries)
        
        # Build the prompt
        prompt = f"""
        You are analyzing multiple scientific reviews to identify common patterns, strengths, weaknesses, and areas for improvement. Your goal is to synthesize these insights to help improve the hypothesis generation process.
        
        Research Goal:
        {research_goal.text}
        
        Reviews Summary:
        {review_text}
        
        Your task is to:
        1. Identify common critiques, issues, or weaknesses across multiple reviews
        2. Identify successful approaches or strengths that appear consistently
        3. Recognize areas where improvements are frequently suggested
        4. Formulate constructive feedback to improve future hypothesis generation
        
        Format your response as a JSON object with the following structure:
        
        ```json
        {{
            "common_issues": ["Issue 1", "Issue 2", ...],
            "successful_approaches": ["Approach 1", "Approach 2", ...],
            "improvement_areas": ["Area 1", "Area 2", ...],
            "synthesis": "Overall synthesis of the review patterns and their implications"
        }}
        ```
        
        Focus on broad patterns rather than specific details of individual reviews.
        """
        
        # Generate meta-review
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
            
            # Create the meta-review
            meta_review = MetaReview(
                research_goal_id=research_goal.id,
                common_issues=data["common_issues"],
                improvement_areas=data["improvement_areas"],
                successful_approaches=data["successful_approaches"]
            )
            
            logger.info(f"Created meta-review for research goal {research_goal.id}")
            return meta_review
            
        except Exception as e:
            logger.error(f"Error parsing meta-review from response: {e}")
            logger.debug(f"Response: {response}")
            
            # Create a basic meta-review in case of error
            meta_review = MetaReview(
                research_goal_id=research_goal.id,
                common_issues=["Error parsing meta-review"],
                improvement_areas=["Error parsing meta-review"],
                successful_approaches=[]
            )
            
            return meta_review
    
    async def analyze_tournament_results(self, 
                                    matches: List[TournamentMatch], 
                                    hypotheses: Dict[str, Hypothesis],
                                    research_goal: ResearchGoal) -> Dict[str, Any]:
        """
        Analyze the results of a tournament.
        
        Args:
            matches (List[TournamentMatch]): The tournament matches.
            hypotheses (Dict[str, Hypothesis]): Dictionary of hypotheses by ID.
            research_goal (ResearchGoal): The research goal.
            
        Returns:
            Dict[str, Any]: The tournament analysis.
        """
        logger.info(f"Analyzing {len(matches)} tournament matches for research goal {research_goal.id}")
        
        # Prepare tournament match summaries
        match_summaries = []
        for i, match in enumerate(matches, 1):
            h1 = hypotheses.get(match.hypothesis1_id)
            h2 = hypotheses.get(match.hypothesis2_id)
            
            if not h1 or not h2:
                continue
                
            winner = hypotheses.get(match.winner_id) if match.winner_id else None
            
            summary = f"Match {i}:\n"
            summary += f"Hypothesis A: {h1.title}\n"
            summary += f"Hypothesis B: {h2.title}\n"
            summary += f"Winner: {'Tie' if not winner else ('A' if winner.id == h1.id else 'B')}\n"
            summary += f"Rationale: {match.rationale[:200]}...\n"
            
            match_summaries.append(summary)
        
        match_text = "\n\n".join(match_summaries)
        
        # Build the prompt
        prompt = f"""
        You are analyzing the results of a tournament that compared scientific hypotheses. Your goal is to identify patterns in which hypotheses tend to win and why.
        
        Research Goal:
        {research_goal.text}
        
        Tournament Match Summaries:
        {match_text}
        
        Your task is to:
        1. Identify key factors that seem to contribute to a hypothesis winning matches
        2. Recognize patterns in the rationales provided for winning hypotheses
        3. Identify common weaknesses in losing hypotheses
        4. Formulate general recommendations for creating strong hypotheses
        
        Format your response as a JSON object with the following structure:
        
        ```json
        {{
            "winning_factors": ["Factor 1", "Factor 2", ...],
            "losing_weaknesses": ["Weakness 1", "Weakness 2", ...],
            "rationale_patterns": ["Pattern 1", "Pattern 2", ...],
            "recommendations": ["Recommendation 1", "Recommendation 2", ...],
            "synthesis": "Overall analysis of tournament results and their implications"
        }}
        ```
        
        Focus on broader patterns rather than specific details of individual matches.
        """
        
        # Generate tournament analysis
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
            
            logger.info(f"Completed tournament analysis for research goal {research_goal.id}")
            return data
            
        except Exception as e:
            logger.error(f"Error parsing tournament analysis from response: {e}")
            logger.debug(f"Response: {response}")
            
            # Return a basic analysis in case of error
            return {
                "winning_factors": ["Error parsing tournament analysis"],
                "losing_weaknesses": ["Error parsing tournament analysis"],
                "rationale_patterns": [],
                "recommendations": [],
                "synthesis": "Error parsing tournament analysis"
            }
    
    async def analyze_protocols(self,
                                  protocols: List[ExperimentalProtocol],
                                  hypotheses: Dict[str, Hypothesis],
                                  research_goal: ResearchGoal) -> Dict[str, Any]:
        """
        Analyze experimental protocols to identify patterns and best practices.
        
        Args:
            protocols (List[ExperimentalProtocol]): The experimental protocols.
            hypotheses (Dict[str, Hypothesis]): Dictionary of hypotheses by ID.
            research_goal (ResearchGoal): The research goal.
            
        Returns:
            Dict[str, Any]: The protocol analysis.
        """
        logger.info(f"Analyzing {len(protocols)} experimental protocols for research goal {research_goal.id}")
        
        if not protocols:
            return {
                "common_elements": [],
                "innovative_approaches": [],
                "methodological_gaps": [],
                "recommendations": []
            }
        
        # Prepare protocol summaries
        protocol_summaries = []
        for i, protocol in enumerate(protocols, 1):
            hypothesis = hypotheses.get(protocol.hypothesis_id)
            if not hypothesis:
                continue
                
            summary = f"Protocol {i} (ID: {protocol.id}):\n"
            summary += f"Title: {protocol.title}\n"
            summary += f"For Hypothesis: {hypothesis.title}\n"
            summary += f"Steps: {len(protocol.steps)}\n"
            summary += f"Materials: {', '.join(protocol.materials[:5])}"
            if len(protocol.materials) > 5:
                summary += f" and {len(protocol.materials) - 5} more"
            summary += f"\nEquipment: {', '.join(protocol.equipment[:3])}"
            if len(protocol.equipment) > 3:
                summary += f" and {len(protocol.equipment) - 3} more"
            summary += f"\nExpected Results: {protocol.expected_results[:100]}...\n"
            
            protocol_summaries.append(summary)
        
        protocol_text = "\n\n".join(protocol_summaries)
        
        # Build the prompt
        prompt = f"""
        You are analyzing multiple experimental protocols designed to test scientific hypotheses. Your goal is to identify common elements, innovative approaches, and potential methodological gaps.
        
        Research Goal:
        {research_goal.text}
        
        Protocol Summaries:
        {protocol_text}
        
        Your task is to:
        1. Identify common methodological elements across protocols
        2. Recognize innovative or unusual experimental approaches
        3. Identify potential methodological gaps or blind spots
        4. Formulate recommendations for designing better protocols
        
        Format your response as a JSON object with the following structure:
        
        ```json
        {{
            "common_elements": ["Element 1", "Element 2", ...],
            "innovative_approaches": ["Approach 1", "Approach 2", ...],
            "methodological_gaps": ["Gap 1", "Gap 2", ...],
            "recommendations": ["Recommendation 1", "Recommendation 2", ...],
            "synthesis": "Overall analysis of the protocols and their implications"
        }}
        ```
        
        Focus on broader patterns rather than specific details of individual protocols.
        """
        
        # Generate protocol analysis
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
            
            logger.info(f"Completed protocol analysis for research goal {research_goal.id}")
            return data
            
        except Exception as e:
            logger.error(f"Error parsing protocol analysis from response: {e}")
            logger.debug(f"Response: {response}")
            
            # Return a basic analysis in case of error
            return {
                "common_elements": ["Error parsing protocol analysis"],
                "innovative_approaches": [],
                "methodological_gaps": ["Error parsing protocol analysis"],
                "recommendations": ["Error parsing protocol analysis"],
                "synthesis": "Error parsing protocol analysis"
            }
    
    async def generate_research_overview(self, 
                                    research_goal: ResearchGoal,
                                    top_hypotheses: List[Hypothesis],
                                    meta_review: Optional[MetaReview] = None,
                                    tournament_analysis: Optional[Dict[str, Any]] = None,
                                    protocol_analysis: Optional[Dict[str, Any]] = None) -> ResearchOverview:
        """
        Generate a research overview based on top hypotheses and insights.
        
        Args:
            research_goal (ResearchGoal): The research goal.
            top_hypotheses (List[Hypothesis]): The top-ranked hypotheses.
            meta_review (Optional[MetaReview]): The meta-review.
            tournament_analysis (Optional[Dict[str, Any]]): The tournament analysis.
            
        Returns:
            ResearchOverview: The research overview.
        """
        logger.info(f"Generating research overview for research goal {research_goal.id} with {len(top_hypotheses)} top hypotheses")
        
        # Prepare hypotheses text
        hypotheses_text = "\n\n".join([
            f"Hypothesis {i+1} (ID: {h.id}):\nTitle: {h.title}\nSummary: {h.summary}\nElo Rating: {h.elo_rating}\nMatches Played: {h.matches_played}"
            for i, h in enumerate(top_hypotheses)
        ])
        
        # Prepare meta-review text
        meta_review_text = ""
        if meta_review:
            meta_review_text = f"""
            Meta-Review Insights:
            
            Common Issues:
            {chr(10).join(['- ' + issue for issue in meta_review.common_issues])}
            
            Improvement Areas:
            {chr(10).join(['- ' + area for area in meta_review.improvement_areas])}
            
            Successful Approaches:
            {chr(10).join(['- ' + approach for approach in meta_review.successful_approaches])}
            """
        
        # Prepare tournament analysis text
        tournament_text = ""
        if tournament_analysis:
            tournament_text = f"""
            Tournament Analysis:
            
            Winning Factors:
            {chr(10).join(['- ' + factor for factor in tournament_analysis.get('winning_factors', [])])}
            
            Common Weaknesses:
            {chr(10).join(['- ' + weakness for weakness in tournament_analysis.get('losing_weaknesses', [])])}
            
            Recommendations:
            {chr(10).join(['- ' + rec for rec in tournament_analysis.get('recommendations', [])])}
            """
            
        # Prepare protocol analysis text
        protocol_text = ""
        if protocol_analysis:
            protocol_text = f"""
            Protocol Analysis:
            
            Common Methodological Elements:
            {chr(10).join(['- ' + element for element in protocol_analysis.get('common_elements', [])])}
            
            Innovative Approaches:
            {chr(10).join(['- ' + approach for approach in protocol_analysis.get('innovative_approaches', [])])}
            
            Methodological Gaps:
            {chr(10).join(['- ' + gap for gap in protocol_analysis.get('methodological_gaps', [])])}
            
            Recommendations:
            {chr(10).join(['- ' + rec for rec in protocol_analysis.get('recommendations', [])])}
            """
        
        # Build the prompt
        prompt = f"""
        You are creating a comprehensive research overview based on the top-ranked hypotheses and insights gained during the research process.
        
        Research Goal:
        {research_goal.text}
        
        Top Hypotheses:
        {hypotheses_text}
        
        {meta_review_text}
        
        {tournament_text}
        
        {protocol_text}
        
        Your task is to:
        1. Synthesize the top hypotheses into a coherent research overview
        2. Identify 3-5 main research areas or directions emerging from these hypotheses
        3. For each research area, suggest specific experiments or investigations
        4. Highlight key open questions and promising directions for future research
        5. Identify potential research contacts with expertise in these areas (these can be fictional or real researchers)
        
        Format your response as a JSON object with the following structure:
        
        ```json
        {{
            "title": "Concise title for the research overview",
            "summary": "Brief executive summary of the research landscape (1-2 paragraphs)",
            "research_areas": [
                {{
                    "name": "Name of research area 1",
                    "description": "Description of this research area",
                    "hypotheses": ["Related hypothesis ID 1", "Related hypothesis ID 2", ...],
                    "experiments": ["Experiment 1", "Experiment 2", ...],
                    "open_questions": ["Question 1", "Question 2", ...]
                }},
                ...
            ],
            "future_directions": ["Direction 1", "Direction 2", ...],
            "potential_contacts": [
                {{
                    "name": "Researcher name",
                    "expertise": "Their expertise",
                    "relevance": "Why they would be relevant to this research"
                }},
                ...
            ]
        }}
        ```
        
        Create a comprehensive and insightful research overview that would help guide future investigations.
        """
        
        # Generate research overview
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
            
            # Create the research overview
            overview = ResearchOverview(
                research_goal_id=research_goal.id,
                title=data["title"],
                summary=data["summary"],
                research_areas=data["research_areas"],
                top_hypotheses=[h.id for h in top_hypotheses],
                potential_contacts=data["potential_contacts"]
            )
            
            logger.info(f"Generated research overview for research goal {research_goal.id}")
            return overview
            
        except Exception as e:
            logger.error(f"Error parsing research overview from response: {e}")
            logger.debug(f"Response: {response}")
            
            # Create a basic overview in case of error
            overview = ResearchOverview(
                research_goal_id=research_goal.id,
                title=f"Research Overview for {research_goal.text[:50]}...",
                summary="Error parsing research overview.",
                research_areas=[{"name": "Error", "description": "Error parsing research overview."}],
                top_hypotheses=[h.id for h in top_hypotheses],
                potential_contacts=[]
            )
            
            return overview
    
    async def format_for_publication(self, 
                                research_overview: ResearchOverview, 
                                format_type: str = "nih_aims") -> str:
        """
        Format the research overview for publication in a specific format.
        
        Args:
            research_overview (ResearchOverview): The research overview.
            format_type (str): The publication format type.
            
        Returns:
            str: The formatted research overview.
        """
        logger.info(f"Formatting research overview {research_overview.id} as {format_type}")
        
        # Build the prompt
        prompt = f"""
        You are formatting a research overview into a publication-ready document following a specific format.
        
        Research Overview:
        Title: {research_overview.title}
        Summary: {research_overview.summary}
        
        Research Areas:
        """
        
        for i, area in enumerate(research_overview.research_areas, 1):
            prompt += f"""
            Research Area {i}: {area.get('name', '')}
            Description: {area.get('description', '')}
            Experiments: {', '.join(area.get('experiments', []))}
            Open Questions: {', '.join(area.get('open_questions', []))}
            """
        
        if format_type == "nih_aims":
            prompt += f"""
            Format this as an NIH Specific Aims page, which should include:
            1. Introduction/background paragraph establishing significance
            2. Knowledge gap/problem statement paragraph
            3. Purpose statement and approach
            4. 2-4 specific aims, each with a clear rationale and approach
            5. Impact statement concluding paragraph
            
            The total document should be no more than 1 page (approximately 500 words). Use appropriate scientific language and formatting expected in an NIH grant proposal.
            """
        elif format_type == "abstract":
            prompt += f"""
            Format this as a scientific abstract for a journal article, which should include:
            1. Background/context
            2. Gap/problem statement
            3. Purpose/approach
            4. Key findings or hypotheses
            5. Implications and significance
            
            The abstract should be approximately 250-300 words and follow conventional scientific abstract structure.
            """
        else:
            prompt += f"""
            Format this as a general research summary for a broad scientific audience, including:
            1. Introduction and context
            2. Key research directions identified
            3. Promising hypotheses
            4. Suggested next steps
            5. Potential impact of the research
            
            The document should be approximately 500-750 words.
            """
        
        # Generate formatted overview
        response = await self.generate(prompt)
        
        logger.info(f"Generated {format_type} formatted overview for research overview {research_overview.id}")
        return response