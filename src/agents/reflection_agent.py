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
        Perform a deep verification review of a hypothesis by decomposing it into sub-assumptions
        and systematically evaluating each component with causal reasoning.
        
        Args:
            hypothesis (Hypothesis): The hypothesis to review.
            research_goal (ResearchGoal): The research goal.
            
        Returns:
            Review: The deep verification review.
        """
        logger.info(f"Performing deep verification review of hypothesis {hypothesis.id}: {hypothesis.title}")
        
        # Perform literature search for verification if available
        literature_context = ""
        if self.literature_search:
            # Search for scientific evidence related to the key components of the hypothesis
            query = f"{hypothesis.title} scientific evidence verification methodology {research_goal.text}"
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
                ## Relevant Scientific Literature for Verification
                Use these sources to inform your verification of the hypothesis components:
                
                {literature_sources}
                """
        
        # Build the prompt
        prompt = f"""
        You are performing a deep verification review of a scientific hypothesis, acting as an expert scientific reviewer with expertise in systematic evaluation and causal reasoning. This involves decomposing the hypothesis into its constituent assumptions, evaluating each component rigorously, and assessing the causal relationships between them.

        Research Goal:
        {research_goal.text}
        
        Hypothesis to Review:
        Title: {hypothesis.title}
        Summary: {hypothesis.summary}
        Description: {hypothesis.description}
        Supporting Evidence: {', '.join(hypothesis.supporting_evidence)}
        
        {literature_context}
        
        Follow these steps for a rigorous scientific verification:
        
        1. Identify the main claim or conclusion of the hypothesis with precision.
        
        2. Decompose the hypothesis systematically:
           - Break down the hypothesis into 4-8 key assumptions or premises that support the main claim
           - Include both explicit assumptions stated in the hypothesis and implicit assumptions necessary for it to hold
           - Identify causal relationships between these assumptions (which ones are causes and which are effects)
        
        3. For each key assumption:
           - Identify 2-5 sub-assumptions or factual claims that must be true for the assumption to hold
           - Assess the logical coherence and connection between sub-assumptions
           - Evaluate for potential logical fallacies or reasoning errors
        
        4. Systematically evaluate each sub-assumption based on:
           - Scientific consensus: Is there strong agreement in the field?
           - Empirical support: What direct evidence exists?
           - Theoretical foundation: Is it consistent with established theories?
           - Methodological soundness: Are studies supporting it well-designed?
        
        5. Classify each sub-assumption as:
           - Well-established (strong scientific consensus and empirical support)
           - Plausible (some evidence but not conclusive, theoretically sound)
           - Speculative (limited evidence, theoretically possible)
           - Controversial (conflicting evidence or theoretical views)
           - Incorrect (contradicts well-established knowledge)
        
        6. For each key assumption, determine:
           - Overall status based on its sub-assumptions
           - Confidence level (high, medium, low) in your assessment
           - Whether it is a central/load-bearing assumption whose failure would invalidate the hypothesis
        
        7. Assess causal reasoning quality:
           - Are cause-effect relationships properly established?
           - Are there confounding variables not addressed?
           - Are correlation-causation errors present?
           - Is the causal chain complete and logically sound?
        
        8. Provide a probabilistic assessment of the hypothesis:
           - Estimate the probability that the hypothesis is correct (0-100%)
           - Identify the specific conditions under which it would be more or less likely to be true
        
        9. Outline concrete experiments or observations that could further verify or falsify the hypothesis.
        
        Format your response as a JSON object with the following structure:
        
        ```json
        {{
            "main_claim": "The precise main claim of the hypothesis...",
            "causal_structure": "Brief description of the causal relationships in the hypothesis...",
            "assumptions": [
                {{
                    "assumption": "First key assumption...",
                    "is_central": true,
                    "sub_assumptions": [
                        {{
                            "sub_assumption": "First sub-assumption...",
                            "status": "well-established/plausible/speculative/controversial/incorrect",
                            "justification": "Your scientific justification with reference to evidence...",
                            "confidence": "high/medium/low"
                        }},
                        ...
                    ],
                    "overall_status": "well-established/plausible/speculative/controversial/incorrect",
                    "confidence": "high/medium/low",
                    "causal_role": "cause/effect/mediator/moderator"
                }},
                ...
            ],
            "causal_reasoning_assessment": "Detailed assessment of the causal reasoning in the hypothesis...",
            "logical_fallacies": ["Fallacy 1 and where it occurs", ...],
            "invalidating_issues": ["Issue 1 that invalidates the hypothesis", ...],
            "fundamental_problems": false,
            "probability_correct": 75,
            "probability_justification": "Justification for the probability estimate...",
            "verification_experiments": ["Experiment 1 that could verify/falsify the hypothesis", ...],
            "overall_assessment": "Your overall assessment with emphasis on scientific rigor..."
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
            
            # Calculate a score based on the assumption statuses and probability assessment
            status_scores = {
                "well-established": 10,
                "plausible": 7.5,
                "speculative": 5,
                "controversial": 3,
                "incorrect": 0
            }
            
            confidence_multipliers = {
                "high": 1.0,
                "medium": 0.8,
                "low": 0.6
            }
            
            # Score each assumption based on its overall status and confidence
            assumption_scores = []
            central_assumption_scores = []
            
            for assumption in data["assumptions"]:
                # Get the status score and confidence multiplier
                status_score = status_scores.get(assumption["overall_status"], 5)
                confidence_mult = confidence_multipliers.get(assumption["confidence"], 0.8)
                
                # Calculate the weighted score
                weighted_score = status_score * confidence_mult
                assumption_scores.append(weighted_score)
                
                # Keep track of scores for central assumptions
                if assumption.get("is_central", False):
                    central_assumption_scores.append(weighted_score)
            
            # Overall score calculation incorporating probability estimate
            probability_factor = data.get("probability_correct", 50) / 50  # normalize to a 0-2 scale
            
            # Calculate basic average from all assumptions
            basic_score = sum(assumption_scores) / len(assumption_scores) if assumption_scores else 5
            
            # If there are central assumptions, they get 2x weight
            if central_assumption_scores:
                central_score = sum(central_assumption_scores) / len(central_assumption_scores)
                overall_score = (basic_score + (2 * central_score)) / 3
            else:
                overall_score = basic_score
                
            # Adjust by probability factor (capped to prevent extreme values)
            probability_adjustment = min(1.5, max(0.5, probability_factor))
            overall_score = overall_score * probability_adjustment
            
            # Severe penalty for fundamental problems
            if data.get("fundamental_problems", False):
                overall_score = max(1, overall_score * 0.4)
                
            # Cap the score at 10
            overall_score = min(10, overall_score)
            
            # Extract strengths and improvement suggestions
            strengths = []
            improvement_suggestions = []
            
            # Consider well-established assumptions as strengths
            for assumption in data["assumptions"]:
                if assumption["overall_status"] in ["well-established", "plausible"] and assumption.get("is_central", False):
                    strengths.append(f"Strong {assumption['overall_status']} central assumption: {assumption['assumption']}")
                    
            # Add verification experiments as improvement suggestions
            for experiment in data.get("verification_experiments", []):
                improvement_suggestions.append(f"Verification experiment: {experiment}")
            
            # Build the review text
            review_text = f"""
            # Deep Verification Review of Hypothesis: {hypothesis.title}
            
            ## Main Claim
            {data["main_claim"]}
            
            ## Causal Structure
            {data.get("causal_structure", "No causal structure provided.")}
            
            ## Assumptions Analysis
            """
            
            for i, assumption in enumerate(data["assumptions"], 1):
                review_text += f"""
                ### Assumption {i}: {assumption["assumption"]}
                **Status: {assumption["overall_status"].upper()}** (Confidence: {assumption["confidence"].upper()})
                **Central to Hypothesis: {"Yes" if assumption.get("is_central", False) else "No"}**
                **Causal Role: {assumption.get("causal_role", "Not specified")}**
                
                #### Sub-assumptions:
                """
                
                for j, sub in enumerate(assumption["sub_assumptions"], 1):
                    review_text += f"""
                    {j}. {sub["sub_assumption"]}
                    - Status: {sub["status"].upper()} (Confidence: {sub.get("confidence", "medium").upper()})
                    - Justification: {sub["justification"]}
                    """
            
            review_text += f"""
            ## Causal Reasoning Assessment
            {data.get("causal_reasoning_assessment", "No causal reasoning assessment provided.")}
            
            ## Logical Fallacies Identified
            {chr(10).join(['- ' + fallacy for fallacy in data.get("logical_fallacies", ["No logical fallacies identified."])])}
            
            ## Invalidating Issues
            {chr(10).join(['- ' + issue for issue in data.get("invalidating_issues", ["No invalidating issues identified."])])}
            
            ## Contains Fundamental Problems
            {"Yes" if data.get("fundamental_problems", False) else "No"}
            
            ## Probability Assessment
            **Estimated Probability of Correctness:** {data.get("probability_correct", "Not provided")}%
            
            **Justification:** {data.get("probability_justification", "No justification provided.")}
            
            ## Verification Experiments
            {chr(10).join(['- ' + exp for exp in data.get("verification_experiments", ["No verification experiments suggested."])])}
            
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
                critiques=data.get("invalidating_issues", []) + data.get("logical_fallacies", []),
                strengths=strengths,
                improvement_suggestions=improvement_suggestions,
                metadata={
                    "probability_correct": data.get("probability_correct", 50),
                    "verification_experiments": data.get("verification_experiments", []),
                    "causal_assessment": data.get("causal_reasoning_assessment", ""),
                    "fundamental_problems": data.get("fundamental_problems", False)
                }
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
        Perform a comprehensive simulation review of a hypothesis using computational modeling
        approach to test predictions and identify sensitivity to initial conditions.
        
        Args:
            hypothesis (Hypothesis): The hypothesis to review.
            research_goal (ResearchGoal): The research goal.
            
        Returns:
            Review: The simulation review.
        """
        logger.info(f"Performing simulation review of hypothesis {hypothesis.id}: {hypothesis.title}")
        
        # Perform literature search for domain-specific methods if available
        literature_context = ""
        if self.literature_search:
            # Search for simulation methods and models related to the hypothesis domain
            query = f"{hypothesis.title} computational model simulation methodology {research_goal.text}"
            search_result = await self.literature_search.search_with_citations(query, max_results=3)
            
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
                ## Relevant Simulation Methodologies from Literature
                Use these sources to inform your simulation approach:
                
                {literature_sources}
                """
        
        # Build the prompt
        prompt = f"""
        You are performing a comprehensive simulation review of a scientific hypothesis, serving as an expert in computational modeling and systems analysis. This review involves creating a detailed computational model of the hypothesis, simulating its behavior under various conditions, and analyzing the outcomes systematically.

        Research Goal:
        {research_goal.text}
        
        Hypothesis to Review:
        Title: {hypothesis.title}
        Summary: {hypothesis.summary}
        Description: {hypothesis.description}
        Supporting Evidence: {', '.join(hypothesis.supporting_evidence)}
        
        {literature_context}
        
        Follow these steps for a rigorous simulation analysis:
        
        1. System Decomposition:
           - Break down the hypothesis into a complete step-by-step process or mechanism
           - Identify all key components, variables, and relationships
           - Determine inputs, outputs, feedback loops, and state variables
        
        2. Model Formulation:
           - Create a formal model structure representing the hypothesis
           - Define governing equations or logical rules for each component
           - Specify parameter values and ranges based on scientific literature
           - Identify initial conditions and boundary conditions
        
        3. Step-by-Step Simulation:
           - For each step in the process, determine:
             * Initial state and inputs
             * Expected outputs according to the hypothesis
             * Dependencies on previous steps
             * Potential alternative pathways
        
        4. Sensitivity Analysis:
           - Identify critical parameters that significantly affect outcomes
           - Determine how robust the model is to parameter variations
           - Assess which initial conditions lead to qualitatively different outcomes
        
        5. Robustness Testing:
           - Introduce perturbations at different points in the simulation
           - Analyze how the system responds to unexpected inputs
           - Identify failure modes and their likelihood
        
        6. Emergent Properties:
           - Identify unexpected behaviors that emerge from the simulation
           - Assess whether these emergent properties support or contradict the hypothesis
           - Determine if the hypothesis can account for known phenomena not explicitly modeled
        
        7. Predictive Power:
           - Generate specific, quantitative predictions from the simulation
           - Compare predictions to known experimental results if available
           - Propose new experiments that could validate simulation predictions
        
        8. Simulation Outcomes:
           - Assess whether the simulation supports, contradicts, or extends the hypothesis
           - Identify unexpected insights generated by the simulation
           - Determine confidence level in the simulation results
        
        Format your response as a JSON object with the following structure:
        
        ```json
        {{
            "model_description": "Formal description of the computational model...",
            "key_components": ["Component 1", "Component 2", ...],
            "key_variables": ["Variable 1", "Variable 2", ...],
            "steps": [
                {{
                    "step_description": "Description of this step in the mechanism...",
                    "inputs": ["Input 1", "Input 2", "..."],
                    "simulation_result": "What happens in this step according to the model...",
                    "output_values": {{
                        "Variable 1": "value1", 
                        "Variable 2": "value2"
                    }},
                    "issues": ["Issue 1 with this step", "Issue 2 with this step", "..."],
                    "confidence": "high/medium/low",
                    "plausibility": "high/medium/low"
                }},
                ...
            ],
            "alternative_pathways": ["Alternative pathway 1", "Alternative pathway 2", ...],
            "failure_points": [
                {{
                    "step_index": 0,
                    "description": "Description of the failure point...",
                    "likelihood": "high/medium/low",
                    "impact": "critical/significant/minor"
                }},
                ...
            ],
            "sensitive_parameters": [
                {{
                    "parameter": "Parameter name...",
                    "sensitivity": "high/medium/low",
                    "critical_values": "Values at which behavior changes significantly..."
                }},
                ...
            ],
            "emergent_properties": ["Emergent property 1", "Emergent property 2", ...],
            "predictions": [
                {{
                    "prediction": "Specific prediction...",
                    "confidence": "high/medium/low",
                    "testable": true
                }},
                ...
            ],
            "overall_plausibility": "high/medium/low",
            "confidence_score": 75,
            "simulation_limitations": ["Limitation 1", "Limitation 2", ...],
            "overall_assessment": "Your overall assessment...",
            "suggested_modifications": [
                {{
                    "modification": "Suggested modification to the hypothesis...",
                    "expected_improvement": "How this would improve the hypothesis...",
                    "implementation_difficulty": "easy/moderate/difficult"
                }},
                ...
            ]
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
            
            # Calculate a score based on plausibility and confidence
            plausibility_scores = {
                "high": 9,
                "medium": 6,
                "low": 3
            }
            
            # Calculate weights for each step based on failure impact
            step_weights = [1.0] * len(data["steps"])  # Default all steps to weight 1.0
            
            # Adjust weights based on failure points
            for failure in data.get("failure_points", []):
                step_idx = failure.get("step_index", 0)
                if 0 <= step_idx < len(step_weights):
                    impact_weight = {
                        "critical": 3.0,
                        "significant": 2.0,
                        "minor": 1.5
                    }.get(failure.get("impact", "minor"), 1.5)
                    
                    likelihood_factor = {
                        "high": 0.2,
                        "medium": 0.5,
                        "low": 0.8
                    }.get(failure.get("likelihood", "medium"), 0.5)
                    
                    # Reduce the weight (more penalty for high likelihood critical failures)
                    step_weights[step_idx] = step_weights[step_idx] * likelihood_factor / impact_weight
            
            # Calculate step scores with weights
            weighted_step_scores = []
            step_plausibilities = [step.get("plausibility", "medium") for step in data["steps"]]
            step_confidences = [step.get("confidence", "medium") for step in data["steps"]]
            
            for i, (plausibility, confidence, weight) in enumerate(zip(step_plausibilities, step_confidences, step_weights)):
                plausibility_score = plausibility_scores.get(plausibility, 5)
                confidence_factor = {"high": 1.0, "medium": 0.85, "low": 0.7}.get(confidence, 0.85)
                weighted_step_scores.append(plausibility_score * confidence_factor * weight)
            
            # Calculate average step score
            avg_step_score = sum(weighted_step_scores) / len(weighted_step_scores) if weighted_step_scores else 5
            
            # Adjust score based on overall plausibility assessment
            overall_plausibility = data.get("overall_plausibility", "medium")
            overall_plausibility_score = plausibility_scores.get(overall_plausibility, 5)
            
            # Factor in confidence score
            confidence_score = data.get("confidence_score", 50)
            confidence_factor = confidence_score / 50  # Normalize to a scale centered at 1.0
            
            # Calculate overall score combining step scores, overall plausibility, and confidence
            combined_score = (avg_step_score * 0.6) + (overall_plausibility_score * 0.4)
            overall_score = combined_score * min(1.5, max(0.5, confidence_factor))
            
            # Apply additional adjustments
            
            # Bonus for many testable predictions
            testable_predictions = [p for p in data.get("predictions", []) if p.get("testable", False)]
            if len(testable_predictions) >= 3:
                overall_score = min(10, overall_score + 0.5)
                
            # Penalty for many sensitive parameters (model that's too brittle)
            high_sensitivity_params = [p for p in data.get("sensitive_parameters", []) if p.get("sensitivity", "") == "high"]
            if len(high_sensitivity_params) >= 3:
                overall_score = max(1, overall_score - 1.0)
                
            # Final cap at 10
            overall_score = min(10, max(1, overall_score))
            
            # Extract strengths, critiques and improvement suggestions
            strengths = []
            critiques = []
            improvement_suggestions = []
            
            # Add emergent properties as strengths
            for prop in data.get("emergent_properties", []):
                strengths.append(f"Emergent property: {prop}")
                
            # Add high-confidence predictions as strengths
            for pred in data.get("predictions", []):
                if pred.get("confidence", "") == "high" and pred.get("testable", False):
                    strengths.append(f"Strong testable prediction: {pred.get('prediction', '')}")
            
            # Add failure points as critiques
            for failure in data.get("failure_points", []):
                critiques.append(f"Failure point: {failure.get('description', '')}")
                
            # Add simulation limitations as critiques
            for limitation in data.get("simulation_limitations", []):
                critiques.append(f"Simulation limitation: {limitation}")
                
            # Add suggested modifications as improvement suggestions
            for suggestion in data.get("suggested_modifications", []):
                improvement_suggestions.append(f"{suggestion.get('modification', '')}: {suggestion.get('expected_improvement', '')}")
            
            # Build the review text
            review_text = f"""
            # Simulation Review of Hypothesis: {hypothesis.title}
            
            ## Computational Model
            {data.get("model_description", "No model description provided.")}
            
            ### Key Components
            {chr(10).join(['- ' + comp for comp in data.get("key_components", ["No key components identified."])])}
            
            ### Key Variables
            {chr(10).join(['- ' + var for var in data.get("key_variables", ["No key variables identified."])])}
            
            ## Step-by-Step Simulation
            """
            
            for i, step in enumerate(data["steps"], 1):
                review_text += f"""
                ### Step {i}: {step["step_description"]}
                **Inputs:** {', '.join(step.get("inputs", ["None specified"]))}
                
                **Simulation Result:** {step["simulation_result"]}
                
                **Output Values:** 
                {chr(10).join(['- ' + key + ': ' + value for key, value in step.get("output_values", {}).items()])}
                
                **Plausibility:** {step.get("plausibility", "medium").upper()} (Confidence: {step.get("confidence", "medium").upper()})
                
                **Issues:**
                {chr(10).join(['- ' + issue for issue in step.get("issues", [])])}
                """
            
            review_text += """
            ## Alternative Pathways
            """
            for pathway in data.get("alternative_pathways", ["No alternative pathways identified."]):
                review_text += f"- {pathway}\n"
            
            review_text += """
            ## Failure Points
            """
            if data.get("failure_points", []):
                for failure in data["failure_points"]:
                    step_index = failure.get("step_index", 0) + 1  # Convert to 1-based for display
                    review_text += f"""
                    - Step {step_index}: {failure.get("description", "")}
                      Likelihood: {failure.get("likelihood", "medium").upper()}, Impact: {failure.get("impact", "minor").upper()}
                    """
            else:
                review_text += "No specific failure points identified.\n"
            
            review_text += """
            ## Sensitivity Analysis
            """
            if data.get("sensitive_parameters", []):
                for param in data["sensitive_parameters"]:
                    review_text += f"""
                    - Parameter: {param.get("parameter", "")}
                      Sensitivity: {param.get("sensitivity", "medium").upper()}
                      Critical Values: {param.get("critical_values", "Not specified")}
                    """
            else:
                review_text += "No sensitivity analysis provided.\n"
            
            review_text += """
            ## Emergent Properties
            """
            for prop in data.get("emergent_properties", ["No emergent properties identified."]):
                review_text += f"- {prop}\n"
            
            review_text += """
            ## Predictions
            """
            if data.get("predictions", []):
                for pred in data["predictions"]:
                    review_text += f"""
                    - {pred.get("prediction", "")}
                      Confidence: {pred.get("confidence", "medium").upper()}, Testable: {"Yes" if pred.get("testable", False) else "No"}
                    """
            else:
                review_text += "No specific predictions generated.\n"
            
            review_text += f"""
            ## Overall Plausibility
            {data.get("overall_plausibility", "medium").upper()} (Confidence: {data.get("confidence_score", 50)}%)
            
            ## Simulation Limitations
            """
            for limitation in data.get("simulation_limitations", ["No limitations specified."]):
                review_text += f"- {limitation}\n"
            
            review_text += f"""
            ## Overall Assessment
            {data["overall_assessment"]}
            
            ## Suggested Modifications
            """
            if data.get("suggested_modifications", []):
                for mod in data["suggested_modifications"]:
                    review_text += f"""
                    - {mod.get("modification", "")}
                      Expected Improvement: {mod.get("expected_improvement", "")}
                      Implementation Difficulty: {mod.get("implementation_difficulty", "moderate").upper()}
                    """
            else:
                review_text += "No specific modifications suggested.\n"
            
            # Create the review
            review = Review(
                hypothesis_id=hypothesis.id,
                review_type=ReviewType.SIMULATION,
                reviewer="reflection",
                text=review_text,
                overall_score=overall_score,
                critiques=critiques,
                strengths=strengths,
                improvement_suggestions=improvement_suggestions,
                metadata={
                    "model_description": data.get("model_description", ""),
                    "predictions": data.get("predictions", []),
                    "confidence_score": data.get("confidence_score", 50),
                    "failure_points": data.get("failure_points", []),
                    "emergent_properties": data.get("emergent_properties", [])
                }
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