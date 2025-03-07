"""
Reflection Agent for reviewing and evaluating hypotheses and experimental protocols.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple

from .base_agent import BaseAgent
from ..config.config import SystemConfig
from ..core.models import Hypothesis, Review, ReviewType, ResearchGoal, Citation, ExperimentalProtocol
from ..tools.web_search import WebSearchTool, ScientificLiteratureSearch

logger = logging.getLogger("co_scientist")

class ReflectionAgent(BaseAgent):
    """
    Agent responsible for reviewing and evaluating hypotheses and experimental protocols.
    
    This agent can:
    1. Perform initial reviews of hypotheses without external tools
    2. Perform comprehensive reviews with literature search integration
    3. Conduct deep verification reviews by decomposing hypotheses into assumptions
    4. Evaluate experimental observations related to hypotheses
    5. Perform step-by-step simulation reviews 
    6. Review experimental protocols for feasibility and scientific rigor
    """
    
    def __init__(self, config: SystemConfig):
        """
        Initialize the Reflection agent.
        
        Args:
            config (SystemConfig): The system configuration.
        """
        super().__init__("reflection", config)
        
        # Initialize web search tools if enabled
        self.web_search = None
        self.literature_search = None
        if config.web_search_enabled:
            # Set provider to "tavily" if available, otherwise use the default
            provider = "tavily" if hasattr(config, "web_search_provider") else "bing"
            api_key = config.web_search_api_key
            
            self.web_search = WebSearchTool(api_key=api_key, provider=provider)
            self.literature_search = ScientificLiteratureSearch(api_key=api_key, provider=provider)
    
    async def initial_review(self, 
                         hypothesis: Hypothesis, 
                         research_goal: ResearchGoal) -> Review:
        """
        Perform an initial review of a hypothesis without external tools.
        
        Args:
            hypothesis (Hypothesis): The hypothesis to review.
            research_goal (ResearchGoal): The research goal.
            
        Returns:
            Review: The initial review.
        """
        logger.info(f"Performing initial review of hypothesis {hypothesis.id}: {hypothesis.title}")
        
        # Build the prompt
        prompt = f"""
        You are performing an initial review of a scientific hypothesis. This review should assess the correctness, quality, novelty, and ethics of the hypothesis WITHOUT using external tools or web searches.

        Research Goal:
        {research_goal.text}
        
        Hypothesis to Review:
        Title: {hypothesis.title}
        Summary: {hypothesis.summary}
        Description: {hypothesis.description}
        Supporting Evidence: {', '.join(hypothesis.supporting_evidence)}
        
        Evaluate the hypothesis on the following criteria:
        1. Correctness: Is the hypothesis scientifically sound and based on accepted principles?
        2. Quality: Is the hypothesis well-formulated, clear, and testable?
        3. Novelty: Does the hypothesis appear to offer new insights or approaches?
        4. Ethics: Are there any ethical concerns with the hypothesis or its potential implementation?
        
        For each criterion, provide a score from 0-10 and a brief justification.
        
        Also provide:
        - Key strengths of the hypothesis
        - Key critiques or weaknesses
        - Suggestions for improvement
        - Overall assessment
        
        Format your response as a JSON object with the following structure:
        
        ```json
        {{
            "correctness_score": 0-10,
            "correctness_justification": "Your justification...",
            "quality_score": 0-10,
            "quality_justification": "Your justification...",
            "novelty_score": 0-10,
            "novelty_justification": "Your justification...",
            "ethics_score": 0-10,
            "ethics_justification": "Your justification...",
            "strengths": ["Strength 1", "Strength 2", ...],
            "critiques": ["Critique 1", "Critique 2", ...],
            "improvement_suggestions": ["Suggestion 1", "Suggestion 2", ...],
            "overall_assessment": "Your overall assessment..."
        }}
        ```
        """
        
        # Generate review
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
            
            # Calculate overall score as average of individual scores
            overall_score = (
                data["correctness_score"] + 
                data["quality_score"] + 
                data["novelty_score"] + 
                data["ethics_score"]
            ) / 4.0
            
            # Build the review text
            review_text = f"""
            # Initial Review of Hypothesis: {hypothesis.title}
            
            ## Correctness (Score: {data["correctness_score"]}/10)
            {data["correctness_justification"]}
            
            ## Quality (Score: {data["quality_score"]}/10)
            {data["quality_justification"]}
            
            ## Novelty (Score: {data["novelty_score"]}/10)
            {data["novelty_justification"]}
            
            ## Ethics (Score: {data["ethics_score"]}/10)
            {data["ethics_justification"]}
            
            ## Key Strengths
            {chr(10).join(['- ' + s for s in data["strengths"]])}
            
            ## Key Critiques
            {chr(10).join(['- ' + c for c in data["critiques"]])}
            
            ## Suggestions for Improvement
            {chr(10).join(['- ' + s for s in data["improvement_suggestions"]])}
            
            ## Overall Assessment
            {data["overall_assessment"]}
            """
            
            # Create the review
            review = Review(
                hypothesis_id=hypothesis.id,
                review_type=ReviewType.INITIAL,
                reviewer="reflection",
                text=review_text,
                novelty_score=data["novelty_score"],
                correctness_score=data["correctness_score"],
                testability_score=data["quality_score"],  # Using quality as proxy for testability
                overall_score=overall_score,
                critiques=data["critiques"],
                strengths=data["strengths"],
                improvement_suggestions=data["improvement_suggestions"]
            )
            
            logger.info(f"Completed initial review of hypothesis {hypothesis.id} with overall score {overall_score:.2f}")
            return review
            
        except Exception as e:
            logger.error(f"Error parsing initial review from response: {e}")
            logger.debug(f"Response: {response}")
            
            # Create a basic review in case of parsing error
            review = Review(
                hypothesis_id=hypothesis.id,
                review_type=ReviewType.INITIAL,
                reviewer="reflection",
                text=f"Error parsing review: {str(e)}\n\nRaw response:\n{response}",
                novelty_score=5.0,
                correctness_score=5.0,
                testability_score=5.0,
                overall_score=5.0,
                critiques=["Error parsing review"],
                strengths=[],
                improvement_suggestions=[]
            )
            
            return review
    
    async def full_review(self, 
                      hypothesis: Hypothesis, 
                      research_goal: ResearchGoal) -> Review:
        """
        Perform a full review of a hypothesis with external tools.
        
        Args:
            hypothesis (Hypothesis): The hypothesis to review.
            research_goal (ResearchGoal): The research goal.
            
        Returns:
            Review: The full review.
        """
        logger.info(f"Performing full review of hypothesis {hypothesis.id}: {hypothesis.title}")
        
        # Check if hypothesis already has citations
        existing_citations = []
        for citation in hypothesis.citations:
            if isinstance(citation, Citation):
                existing_citations.append(citation)
            elif isinstance(citation, str):
                # We would need to fetch the citation object from database
                # This would be implemented in a real system
                pass
        
        # Prepare citation context from existing citations
        citation_context = ""
        if existing_citations:
            citation_list = "\n".join([
                f"- {citation.title} - {citation.url}"
                for citation in existing_citations
            ])
            citation_context = f"\n## Existing Citations in Hypothesis\n{citation_list}\n"
                    
        # Perform literature search if scientific literature search is enabled
        literature_context = ""
        search_results = []
        citations = []
        
        if self.literature_search:
            # Prepare literature search query based on the hypothesis
            query = f"{hypothesis.title} {research_goal.text} scientific research"
            
            # Use the scientific literature search
            search_result = await self.literature_search.search_with_citations(query, max_results=5)
            
            # Extract results and citations
            search_results = search_result.get("results", [])
            citations = search_result.get("citations", [])
            
            if search_results:
                # Format the literature context
                literature_sources = "\n\n".join([
                    f"Source {i+1}: {result.get('title', 'Untitled')}\n"
                    f"URL: {result.get('url', 'No URL')}\n"
                    f"Summary: {result.get('snippet', 'No snippet available')}"
                    for i, result in enumerate(search_results)
                ])
                
                # Create a citation reference guide
                citation_guide = "\n".join([
                    f"[{i+1}] {citation.get('title', 'Untitled')} - {citation.get('url', 'No URL')}"
                    for i, citation in enumerate(citations)
                ])
                
                literature_context = f"""
                ## Relevant Scientific Literature
                {literature_sources}
                
                ## Available Citations
                {citation_guide}
                """
        
        # Build the prompt
        prompt = f"""
        You are performing a comprehensive review of a scientific hypothesis, using available literature and scientific knowledge.

        Research Goal:
        {research_goal.text}
        
        Hypothesis to Review:
        Title: {hypothesis.title}
        Summary: {hypothesis.summary}
        Description: {hypothesis.description}
        Supporting Evidence: {', '.join(hypothesis.supporting_evidence)}
        
        {citation_context}
        {literature_context}
        
        Evaluate the hypothesis thoroughly on the following criteria:
        1. Correctness: Is the hypothesis scientifically sound? Are the underlying assumptions valid? Is it consistent with established scientific knowledge?
        2. Quality: Is the hypothesis well-formulated, specific, and testable? Does it make precise predictions?
        3. Novelty: Is this hypothesis truly novel? Does it extend beyond what is already known in the field? Cite specific literature if the hypothesis (or parts of it) have been previously proposed.
        4. Testability: Can the hypothesis be tested with current technology and methods? What experiments would be needed?
        5. Literature Grounding: Is the hypothesis well-grounded in existing literature? Does it cite relevant sources?
        
        For each criterion, provide a score from a 0-10 scale and a detailed justification based on scientific principles and literature.
        
        Also provide:
        - Key strengths of the hypothesis
        - Key critiques or weaknesses
        - Specific suggestions for improvement
        - Overall assessment and recommendation (accept, revise, or reject)
        
        Format your response as a JSON object with the following structure:
        
        ```json
        {{
            "correctness_score": 0-10,
            "correctness_justification": "Your detailed justification...",
            "quality_score": 0-10,
            "quality_justification": "Your detailed justification...",
            "novelty_score": 0-10,
            "novelty_justification": "Your detailed justification...",
            "testability_score": 0-10,
            "testability_justification": "Your detailed justification...",
            "literature_grounding_score": 0-10,
            "literature_grounding_justification": "Your detailed justification...",
            "strengths": ["Strength 1", "Strength 2", ...],
            "critiques": ["Critique 1", "Critique 2", ...],
            "improvement_suggestions": ["Suggestion 1", "Suggestion 2", ...],
            "overall_assessment": "Your detailed overall assessment...",
            "recommendation": "accept", "revise", or "reject",
            "citation_ids": [1, 2, 3],
            "literature_references": ["Reference 1", "Reference 2", ...]
        }}
        ```
        
        If literature was provided, include the citation_ids of the most relevant citations (by number) that either support or refute the hypothesis.
        """
        
        # Generate review
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
            
            # Calculate overall score as average of individual scores
            overall_score = (
                data["correctness_score"] + 
                data["quality_score"] + 
                data["novelty_score"] + 
                data["testability_score"] + 
                data.get("literature_grounding_score", 5)  # Default to 5 if not provided
            ) / 5.0
            
            # Process citations if available
            review_citations = []
            citation_ids = data.get("citation_ids", [])
            
            if isinstance(citation_ids, list) and len(citations) > 0:
                for cid in citation_ids:
                    # Convert citation ID to 0-based index
                    idx = int(cid) - 1
                    if 0 <= idx < len(citations):
                        citation_info = citations[idx]
                        citation = Citation(
                            title=citation_info.get("title", "Unknown"),
                            url=citation_info.get("url", ""),
                            authors=[],
                            snippet=citation_info.get("snippet", ""),
                            source="review_literature_search",
                            publication_date=citation_info.get("publication_date", ""),
                            metadata={
                                "hypothesis_id": hypothesis.id,
                                "citation_index": cid,
                                "relevance": "review_citation"
                            }
                        )
                        review_citations.append(citation)
            
            # Format cited references
            cited_refs = []
            for i, cid in enumerate(citation_ids):
                idx = int(cid) - 1
                if 0 <= idx < len(citations):
                    citation_info = citations[idx]
                    cited_refs.append(f"[{cid}] {citation_info.get('title', 'Unknown')} - {citation_info.get('url', '')}")
            
            # Add literature references that aren't in the citations
            for ref in data.get("literature_references", []):
                if not any(ref in cited_ref for cited_ref in cited_refs):
                    cited_refs.append(ref)
            
            # Build the review text
            review_text = f"""
            # Full Review of Hypothesis: {hypothesis.title}
            
            ## Correctness (Score: {data["correctness_score"]}/10)
            {data["correctness_justification"]}
            
            ## Quality (Score: {data["quality_score"]}/10)
            {data["quality_justification"]}
            
            ## Novelty (Score: {data["novelty_score"]}/10)
            {data["novelty_justification"]}
            
            ## Testability (Score: {data["testability_score"]}/10)
            {data["testability_justification"]}
            
            ## Literature Grounding (Score: {data.get("literature_grounding_score", 5)}/10)
            {data.get("literature_grounding_justification", "No assessment provided.")}
            
            ## Key Strengths
            {chr(10).join(['- ' + s for s in data["strengths"]])}
            
            ## Key Critiques
            {chr(10).join(['- ' + c for c in data["critiques"]])}
            
            ## Suggestions for Improvement
            {chr(10).join(['- ' + s for s in data["improvement_suggestions"]])}
            
            ## Overall Assessment
            {data["overall_assessment"]}
            
            ## Recommendation
            {data["recommendation"].upper()}
            
            ## Literature References
            {chr(10).join(['- ' + r for r in cited_refs])}
            """
            
            # Create the review with citations
            review = Review(
                hypothesis_id=hypothesis.id,
                review_type=ReviewType.FULL,
                reviewer="reflection",
                text=review_text,
                novelty_score=data["novelty_score"],
                correctness_score=data["correctness_score"],
                testability_score=data["testability_score"],
                overall_score=overall_score,
                critiques=data["critiques"],
                strengths=data["strengths"],
                improvement_suggestions=data["improvement_suggestions"],
                metadata={
                    "literature_grounding_score": data.get("literature_grounding_score", 5),
                    "citation_ids": citation_ids
                }
            )
            
            logger.info(f"Completed full review of hypothesis {hypothesis.id} with overall score {overall_score:.2f}")
            return review
            
        except Exception as e:
            logger.error(f"Error parsing full review from response: {e}")
            logger.debug(f"Response: {response}")
            
            # Create a basic review in case of parsing error
            review = Review(
                hypothesis_id=hypothesis.id,
                review_type=ReviewType.FULL,
                reviewer="reflection",
                text=f"Error parsing review: {str(e)}\n\nRaw response:\n{response}",
                novelty_score=5.0,
                correctness_score=5.0,
                testability_score=5.0,
                overall_score=5.0,
                critiques=["Error parsing review"],
                strengths=[],
                improvement_suggestions=[]
            )
            
            return review
    
    async def deep_verification_review(self, 
                                   hypothesis: Hypothesis, 
                                   research_goal: ResearchGoal) -> Review:
        """
        Perform a deep verification review of a hypothesis by decomposing it into sub-assumptions.
        
        Args:
            hypothesis (Hypothesis): The hypothesis to review.
            research_goal (ResearchGoal): The research goal.
            
        Returns:
            Review: The deep verification review.
        """
        logger.info(f"Performing deep verification review of hypothesis {hypothesis.id}: {hypothesis.title}")
        
        # Build the prompt
        prompt = f"""
        You are performing a deep verification review of a scientific hypothesis. This involves decomposing the hypothesis into its constituent assumptions and evaluating each one independently.

        Research Goal:
        {research_goal.text}
        
        Hypothesis to Review:
        Title: {hypothesis.title}
        Summary: {hypothesis.summary}
        Description: {hypothesis.description}
        Supporting Evidence: {', '.join(hypothesis.supporting_evidence)}
        
        Follow these steps:
        1. Identify the main claim or conclusion of the hypothesis.
        2. Break down the hypothesis into 3-7 key assumptions or premises that support the main claim.
        3. For each assumption, identify 2-4 sub-assumptions or facts that must be true for the assumption to hold.
        4. Evaluate each sub-assumption independently, indicating whether it is:
           - Well-established (strong scientific consensus)
           - Plausible (some evidence but not conclusive)
           - Speculative (limited or no evidence)
           - Incorrect (contradicts established knowledge)
        5. Determine whether any incorrect assumptions are fundamental to the hypothesis (would invalidate it if wrong).
        6. Provide an overall assessment of the hypothesis based on this decomposition.
        
        Format your response as a JSON object with the following structure:
        
        ```json
        {{
            "main_claim": "The main claim or conclusion of the hypothesis...",
            "assumptions": [
                {{
                    "assumption": "First key assumption...",
                    "sub_assumptions": [
                        {{
                            "sub_assumption": "First sub-assumption...",
                            "status": "well-established", "plausible", "speculative", or "incorrect",
                            "justification": "Your justification..."
                        }},
                        ...
                    ],
                    "overall_status": "well-established", "plausible", "speculative", or "incorrect"
                }},
                ...
            ],
            "invalidating_issues": ["Issue 1 that invalidates the hypothesis", ...],
            "fundamental_problems": true or false,
            "overall_assessment": "Your overall assessment..."
        }}
        ```
        """
        
        # Generate review
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
            
            # Calculate a score based on the assumption statuses
            status_scores = {
                "well-established": 10,
                "plausible": 7,
                "speculative": 4,
                "incorrect": 0
            }
            
            assumption_scores = []
            for assumption in data["assumptions"]:
                sub_scores = [status_scores.get(sub["status"], 5) for sub in assumption["sub_assumptions"]]
                avg_sub_score = sum(sub_scores) / len(sub_scores) if sub_scores else 5
                assumption_scores.append(avg_sub_score)
            
            overall_score = sum(assumption_scores) / len(assumption_scores) if assumption_scores else 5
            if data.get("fundamental_problems", False):
                overall_score = max(2, overall_score - 5)  # Penalty for fundamental problems
            
            # Build the review text
            review_text = f"""
            # Deep Verification Review of Hypothesis: {hypothesis.title}
            
            ## Main Claim
            {data["main_claim"]}
            
            ## Assumptions Analysis
            """
            
            for i, assumption in enumerate(data["assumptions"], 1):
                review_text += f"""
                ### Assumption {i}: {assumption["assumption"]}
                **Status: {assumption["overall_status"].upper()}**
                
                #### Sub-assumptions:
                """
                
                for j, sub in enumerate(assumption["sub_assumptions"], 1):
                    review_text += f"""
                    {j}. {sub["sub_assumption"]}
                    - Status: {sub["status"].upper()}
                    - Justification: {sub["justification"]}
                    """
            
            review_text += f"""
            ## Invalidating Issues
            {chr(10).join(['- ' + issue for issue in data.get("invalidating_issues", [])])}
            
            ## Contains Fundamental Problems
            {"Yes" if data.get("fundamental_problems", False) else "No"}
            
            ## Overall Assessment
            {data["overall_assessment"]}
            """
            
            # Create the review
            review = Review(
                hypothesis_id=hypothesis.id,
                review_type=ReviewType.DEEP_VERIFICATION,
                reviewer="reflection",
                text=review_text,
                correctness_score=overall_score,
                overall_score=overall_score,
                critiques=data.get("invalidating_issues", []),
                strengths=[],
                improvement_suggestions=[]
            )
            
            logger.info(f"Completed deep verification review of hypothesis {hypothesis.id} with overall score {overall_score:.2f}")
            return review
            
        except Exception as e:
            logger.error(f"Error parsing deep verification review from response: {e}")
            logger.debug(f"Response: {response}")
            
            # Create a basic review in case of parsing error
            review = Review(
                hypothesis_id=hypothesis.id,
                review_type=ReviewType.DEEP_VERIFICATION,
                reviewer="reflection",
                text=f"Error parsing review: {str(e)}\n\nRaw response:\n{response}",
                correctness_score=5.0,
                overall_score=5.0,
                critiques=["Error parsing review"],
                strengths=[],
                improvement_suggestions=[]
            )
            
            return review
    
    async def observation_review(self, 
                            hypothesis: Hypothesis, 
                            research_goal: ResearchGoal) -> Review:
        """
        Review whether a hypothesis can explain existing experimental observations.
        
        Args:
            hypothesis (Hypothesis): The hypothesis to review.
            research_goal (ResearchGoal): The research goal.
            
        Returns:
            Review: The observation review.
        """
        logger.info(f"Performing observation review of hypothesis {hypothesis.id}: {hypothesis.title}")
        
        # Perform literature search if scientific literature search is enabled
        literature_context = ""
        if self.literature_search:
            # Search for experimental observations related to the hypothesis
            query = f"{hypothesis.title} {research_goal.text} experimental evidence observations"
            search_result = await self.literature_search.search_with_citations(query, max_results=5)
            
            # Extract results and format them for the prompt
            search_results = search_result.get("results", [])
            
            if search_results:
                # Format the literature context
                literature_sources = "\n\n".join([
                    f"Source {i+1}: {result.get('title', 'Untitled')}\n"
                    f"URL: {result.get('url', 'No URL')}\n"
                    f"Summary: {result.get('snippet', 'No snippet available')}"
                    for i, result in enumerate(search_results)
                ])
                
                literature_context = f"""
                ## Relevant Experimental Literature
                Use these sources to inform your thinking about relevant experimental observations:
                
                {literature_sources}
                """
        
        # Build the prompt
        prompt = f"""
        You are evaluating whether a scientific hypothesis can account for existing experimental observations and phenomena in the literature.

        Research Goal:
        {research_goal.text}
        
        Hypothesis to Review:
        Title: {hypothesis.title}
        Summary: {hypothesis.summary}
        Description: {hypothesis.description}
        Supporting Evidence: {', '.join(hypothesis.supporting_evidence)}
        
        {literature_context}
        
        Your task is to:
        1. Think of 3-5 important experimental observations or phenomena from the scientific literature that are relevant to this hypothesis and research goal.
        2. For each observation, assess whether the hypothesis provides a better explanation than current theories.
        3. Provide a detailed justification for each assessment.
        4. Determine whether any of these observations strongly support or refute the hypothesis.
        
        Format your response as a JSON object with the following structure:
        
        ```json
        {{
            "observations": [
                {{
                    "observation": "Description of the observation or phenomenon...",
                    "source": "Source of the observation (paper, experiment, etc.)...",
                    "current_explanation": "Current scientific explanation...",
                    "hypothesis_explanation": "How the hypothesis explains this observation...",
                    "is_better_explanation": true or false,
                    "justification": "Your justification..."
                }},
                ...
            ],
            "supporting_observations": ["Index 0-based of supporting observations"],
            "refuting_observations": ["Index 0-based of refuting observations"],
            "overall_assessment": "Your overall assessment..."
        }}
        ```
        """
        
        # Generate review
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
            
            # Calculate a score based on the observations
            num_observations = len(data["observations"])
            num_supporting = len(data.get("supporting_observations", []))
            num_refuting = len(data.get("refuting_observations", []))
            
            if num_observations == 0:
                overall_score = 5.0  # Neutral if no observations
            else:
                # Score is proportion of supporting observations (0-10 scale)
                overall_score = (num_supporting / num_observations) * 10
                # Penalty for refuting observations
                if num_refuting > 0:
                    overall_score = max(0, overall_score - (num_refuting / num_observations) * 5)
            
            # Build the review text
            review_text = f"""
            # Observation Review of Hypothesis: {hypothesis.title}
            
            ## Relevant Observations from Literature
            """
            
            for i, obs in enumerate(data["observations"]):
                review_text += f"""
                ### Observation {i+1}: {obs["observation"]}
                **Source:** {obs["source"]}
                
                **Current Explanation:** {obs["current_explanation"]}
                
                **Hypothesis Explanation:** {obs["hypothesis_explanation"]}
                
                **Is Better Explanation:** {"Yes" if obs.get("is_better_explanation", False) else "No"}
                
                **Justification:** {obs["justification"]}
                """
            
            review_text += f"""
            ## Supporting Observations
            {chr(10).join(['- ' + data["observations"][int(i)]["observation"] for i in data.get("supporting_observations", [])])}
            
            ## Refuting Observations
            {chr(10).join(['- ' + data["observations"][int(i)]["observation"] for i in data.get("refuting_observations", [])])}
            
            ## Overall Assessment
            {data["overall_assessment"]}
            """
            
            # Create critiques and strengths
            critiques = [data["observations"][int(i)]["observation"] for i in data.get("refuting_observations", [])]
            strengths = [data["observations"][int(i)]["observation"] for i in data.get("supporting_observations", [])]
            
            # Create the review
            review = Review(
                hypothesis_id=hypothesis.id,
                review_type=ReviewType.OBSERVATION,
                reviewer="reflection",
                text=review_text,
                overall_score=overall_score,
                critiques=critiques,
                strengths=strengths,
                improvement_suggestions=[]
            )
            
            logger.info(f"Completed observation review of hypothesis {hypothesis.id} with overall score {overall_score:.2f}")
            return review
            
        except Exception as e:
            logger.error(f"Error parsing observation review from response: {e}")
            logger.debug(f"Response: {response}")
            
            # Create a basic review in case of parsing error
            review = Review(
                hypothesis_id=hypothesis.id,
                review_type=ReviewType.OBSERVATION,
                reviewer="reflection",
                text=f"Error parsing review: {str(e)}\n\nRaw response:\n{response}",
                overall_score=5.0,
                critiques=["Error parsing review"],
                strengths=[],
                improvement_suggestions=[]
            )
            
            return review
            
    async def simulation_review(self, 
                           hypothesis: Hypothesis, 
                           research_goal: ResearchGoal) -> Review:
        """
        Perform a step-by-step simulation review of a hypothesis.
        
        Args:
            hypothesis (Hypothesis): The hypothesis to review.
            research_goal (ResearchGoal): The research goal.
            
        Returns:
            Review: The simulation review.
        """
        logger.info(f"Performing simulation review of hypothesis {hypothesis.id}: {hypothesis.title}")
        
        # Build the prompt
        prompt = f"""
        You are performing a simulation review of a scientific hypothesis. This involves simulating the hypothesis step-by-step to identify potential failure points and inconsistencies.

        Research Goal:
        {research_goal.text}
        
        Hypothesis to Review:
        Title: {hypothesis.title}
        Summary: {hypothesis.summary}
        Description: {hypothesis.description}
        Supporting Evidence: {', '.join(hypothesis.supporting_evidence)}
        
        Follow these steps:
        1. Break down the hypothesis into a step-by-step model or mechanism.
        2. For each step, simulate what would happen according to the hypothesis.
        3. Identify any points where the simulation reveals inconsistencies, implausibilities, or contradictions.
        4. Assess whether the overall simulation supports or refutes the hypothesis.
        
        Format your response as a JSON object with the following structure:
        
        ```json
        {{
            "steps": [
                {{
                    "step_description": "Description of this step in the mechanism...",
                    "simulation_result": "What happens in this step according to the hypothesis...",
                    "issues": ["Issue 1 with this step", "Issue 2 with this step", ...],
                    "plausibility": "high", "medium", or "low"
                }},
                ...
            ],
            "failure_points": ["Step index (0-based) of failure points"],
            "overall_plausibility": "high", "medium", or "low",
            "overall_assessment": "Your overall assessment...",
            "suggestions": ["Suggestion 1 to improve the hypothesis", ...]
        }}
        ```
        """
        
        # Generate review
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
            
            # Calculate a score based on the plausibility
            plausibility_scores = {
                "high": 9,
                "medium": 6,
                "low": 3
            }
            
            # Count steps with different plausibility levels
            step_plausibilities = [step.get("plausibility", "medium") for step in data["steps"]]
            
            # Calculate overall score based on individual step plausibilities
            step_scores = [plausibility_scores.get(p, 5) for p in step_plausibilities]
            avg_step_score = sum(step_scores) / len(step_scores) if step_scores else 5
            
            # Adjust score based on overall plausibility assessment
            overall_plausibility = data.get("overall_plausibility", "medium")
            overall_plausibility_score = plausibility_scores.get(overall_plausibility, 5)
            
            # Combine step average and overall assessment
            overall_score = (avg_step_score + overall_plausibility_score) / 2
            
            # Apply penalty for failure points
            num_failures = len(data.get("failure_points", []))
            if num_failures > 0:
                failure_penalty = min(4, num_failures)  # Cap penalty at 4 points
                overall_score = max(1, overall_score - failure_penalty)
            
            # Build the review text
            review_text = f"""
            # Simulation Review of Hypothesis: {hypothesis.title}
            
            ## Step-by-Step Simulation
            """
            
            for i, step in enumerate(data["steps"], 1):
                review_text += f"""
                ### Step {i}: {step["step_description"]}
                **Simulation Result:** {step["simulation_result"]}
                
                **Plausibility:** {step.get("plausibility", "medium").upper()}
                
                **Issues:**
                {chr(10).join(['- ' + issue for issue in step.get("issues", [])])}
                """
            
            review_text += f"""
            ## Failure Points
            {chr(10).join(['- Step ' + str(int(i) + 1) + ': ' + data["steps"][int(i)]["step_description"] for i in data.get("failure_points", [])])}
            
            ## Overall Plausibility
            {data.get("overall_plausibility", "medium").upper()}
            
            ## Overall Assessment
            {data["overall_assessment"]}
            
            ## Suggestions for Improvement
            {chr(10).join(['- ' + s for s in data.get("suggestions", [])])}
            """
            
            # Create critiques and improvement suggestions
            critiques = []
            for i in data.get("failure_points", []):
                i = int(i)
                if i < len(data["steps"]):
                    critiques.extend(data["steps"][i].get("issues", []))
            
            # Create the review
            review = Review(
                hypothesis_id=hypothesis.id,
                review_type=ReviewType.SIMULATION,
                reviewer="reflection",
                text=review_text,
                overall_score=overall_score,
                critiques=critiques,
                strengths=[],
                improvement_suggestions=data.get("suggestions", [])
            )
            
            logger.info(f"Completed simulation review of hypothesis {hypothesis.id} with overall score {overall_score:.2f}")
            return review
            
        except Exception as e:
            logger.error(f"Error parsing simulation review from response: {e}")
            logger.debug(f"Response: {response}")
            
            # Create a basic review in case of parsing error
            review = Review(
                hypothesis_id=hypothesis.id,
                review_type=ReviewType.SIMULATION,
                reviewer="reflection",
                text=f"Error parsing review: {str(e)}\n\nRaw response:\n{response}",
                overall_score=5.0,
                critiques=["Error parsing review"],
                strengths=[],
                improvement_suggestions=[]
            )
            
            return review
            
    async def review_protocol(self, 
                        protocol: ExperimentalProtocol, 
                        hypothesis: Hypothesis,
                        research_goal: ResearchGoal) -> Review:
        """
        Review an experimental protocol for scientific rigor, feasibility, and alignment with the hypothesis.
        
        Args:
            protocol (ExperimentalProtocol): The protocol to review.
            hypothesis (Hypothesis): The hypothesis the protocol aims to test.
            research_goal (ResearchGoal): The research goal.
            
        Returns:
            Review: The protocol review.
        """
        logger.info(f"Reviewing experimental protocol {protocol.id} for hypothesis {hypothesis.id}")
        
        # Perform literature search if scientific literature search is enabled
        literature_context = ""
        if self.literature_search:
            # Search for experimental methods related to the protocol
            query = f"{protocol.title} experimental methods techniques {hypothesis.title}"
            search_result = await self.literature_search.search_with_citations(query, max_results=3)
            
            # Extract results and citations
            search_results = search_result.get("results", [])
            
            if search_results:
                # Format the literature context
                literature_sources = "\n\n".join([
                    f"Source {i+1}: {result.get('title', 'Untitled')}\n"
                    f"URL: {result.get('url', 'No URL')}\n"
                    f"Summary: {result.get('snippet', 'No snippet available')}"
                    for i, result in enumerate(search_results)
                ])
                
                literature_context = f"""
                ## Relevant Experimental Literature
                Use these sources to inform your assessment of the protocol:
                
                {literature_sources}
                """
        
        # Build the prompt
        prompt = f"""
        You are reviewing an experimental protocol designed to test a scientific hypothesis. Your task is to evaluate 
        the protocol for scientific rigor, feasibility, and its ability to effectively test the hypothesis.

        Research Goal:
        {research_goal.text}
        
        Hypothesis to Test:
        Title: {hypothesis.title}
        Summary: {hypothesis.summary}
        Description: {hypothesis.description}
        
        Experimental Protocol to Review:
        Title: {protocol.title}
        Description: {protocol.description}
        
        Steps:
        {chr(10).join(['- ' + step for step in protocol.steps])}
        
        Materials:
        {chr(10).join(['- ' + material for material in protocol.materials])}
        
        Equipment:
        {chr(10).join(['- ' + equipment for equipment in protocol.equipment])}
        
        Expected Results:
        {protocol.expected_results}
        
        Limitations:
        {chr(10).join(['- ' + limitation for limitation in protocol.limitations])}
        
        {literature_context}
        
        Please evaluate the protocol on the following criteria:
        1. Scientific Rigor: Does the protocol follow sound scientific principles? Does it include proper controls?
        2. Feasibility: Is the protocol practically feasible? Are the methods, materials, and equipment readily available?
        3. Alignment: Does the protocol directly test the key aspects of the hypothesis?
        4. Clarity: Are the steps clear, detailed, and reproducible?
        5. Analysis: Is the approach to data analysis appropriate?
        
        For each criterion, provide a score from 0-10 and a detailed justification.
        
        Also provide:
        - Key strengths of the protocol
        - Areas for improvement
        - Specific suggestions to enhance the protocol
        - Overall assessment
        
        Format your response as a JSON object with the following structure:
        
        ```json
        {{
            "rigor_score": 0-10,
            "rigor_justification": "Your justification...",
            "feasibility_score": 0-10,
            "feasibility_justification": "Your justification...",
            "alignment_score": 0-10,
            "alignment_justification": "Your justification...",
            "clarity_score": 0-10,
            "clarity_justification": "Your justification...",
            "analysis_score": 0-10,
            "analysis_justification": "Your justification...",
            "strengths": ["Strength 1", "Strength 2", ...],
            "areas_for_improvement": ["Area 1", "Area 2", ...],
            "suggestions": ["Suggestion 1", "Suggestion 2", ...],
            "overall_assessment": "Your overall assessment..."
        }}
        ```
        """
        
        # Generate review
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
            
            # Calculate overall score as average of individual scores
            overall_score = (
                data["rigor_score"] + 
                data["feasibility_score"] + 
                data["alignment_score"] + 
                data["clarity_score"] + 
                data["analysis_score"]
            ) / 5.0
            
            # Build the review text
            review_text = f"""
            # Review of Experimental Protocol: {protocol.title}
            
            ## Scientific Rigor (Score: {data["rigor_score"]}/10)
            {data["rigor_justification"]}
            
            ## Feasibility (Score: {data["feasibility_score"]}/10)
            {data["feasibility_justification"]}
            
            ## Alignment with Hypothesis (Score: {data["alignment_score"]}/10)
            {data["alignment_justification"]}
            
            ## Clarity (Score: {data["clarity_score"]}/10)
            {data["clarity_justification"]}
            
            ## Analysis Approach (Score: {data["analysis_score"]}/10)
            {data["analysis_justification"]}
            
            ## Key Strengths
            {chr(10).join(['- ' + s for s in data["strengths"]])}
            
            ## Areas for Improvement
            {chr(10).join(['- ' + a for a in data["areas_for_improvement"]])}
            
            ## Suggestions
            {chr(10).join(['- ' + s for s in data["suggestions"]])}
            
            ## Overall Assessment
            {data["overall_assessment"]}
            """
            
            # Create the review
            review = Review(
                hypothesis_id=hypothesis.id,  # Link to the hypothesis being tested
                review_type=ReviewType.FULL,  # Using FULL type for protocol reviews
                reviewer="reflection",
                text=review_text,
                overall_score=overall_score,
                critiques=data["areas_for_improvement"],
                strengths=data["strengths"],
                improvement_suggestions=data["suggestions"],
                metadata={
                    "protocol_id": protocol.id,
                    "rigor_score": data["rigor_score"],
                    "feasibility_score": data["feasibility_score"],
                    "alignment_score": data["alignment_score"],
                    "clarity_score": data["clarity_score"],
                    "analysis_score": data["analysis_score"],
                    "review_target": "protocol"
                }
            )
            
            logger.info(f"Completed review of protocol {protocol.id} with overall score {overall_score:.2f}")
            return review
            
        except Exception as e:
            logger.error(f"Error parsing protocol review from response: {e}")
            logger.debug(f"Response: {response}")
            
            # Create a basic review in case of parsing error
            review = Review(
                hypothesis_id=hypothesis.id,
                review_type=ReviewType.FULL,
                reviewer="reflection",
                text=f"Error parsing protocol review: {str(e)}\n\nRaw response:\n{response}",
                overall_score=5.0,
                critiques=["Error parsing review"],
                strengths=[],
                improvement_suggestions=[],
                metadata={"protocol_id": protocol.id, "review_target": "protocol"}
            )
            
            return review