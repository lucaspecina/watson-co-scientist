"""
Reflection Agent for reviewing and evaluating hypotheses.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple

from .base_agent import BaseAgent
from ..config.config import SystemConfig
from ..core.models import Hypothesis, Review, ReviewType, ResearchGoal
from ..tools.web_search import WebSearchTool

logger = logging.getLogger("co_scientist")

class ReflectionAgent(BaseAgent):
    """
    Agent responsible for reviewing and evaluating hypotheses.
    """
    
    def __init__(self, config: SystemConfig):
        """
        Initialize the Reflection agent.
        
        Args:
            config (SystemConfig): The system configuration.
        """
        super().__init__("reflection", config)
        
        # Initialize web search tool if enabled
        self.web_search = None
        if config.web_search_enabled:
            self.web_search = WebSearchTool(config.web_search_api_key)
    
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
        
        # Perform literature search if web search is enabled
        literature_context = ""
        if self.web_search:
            # Search for literature related to the hypothesis
            query = f"{hypothesis.title} {research_goal.text} scientific research"
            search_results = await self.web_search.search(query, count=5)
            
            if search_results:
                literature_context = "## Relevant Literature\n\n" + "\n\n".join([
                    f"Title: {result['title']}\nURL: {result['url']}\nSummary: {result['snippet']}"
                    for result in search_results
                ])
        
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
        
        {literature_context}
        
        Evaluate the hypothesis thoroughly on the following criteria:
        1. Correctness: Is the hypothesis scientifically sound? Are the underlying assumptions valid? Is it consistent with established scientific knowledge?
        2. Quality: Is the hypothesis well-formulated, specific, and testable? Does it make precise predictions?
        3. Novelty: Is this hypothesis truly novel? Does it extend beyond what is already known in the field? Cite specific literature if the hypothesis (or parts of it) have been previously proposed.
        4. Testability: Can the hypothesis be tested with current technology and methods? What experiments would be needed?
        5. Ethics: Are there any ethical concerns with the hypothesis or its potential implementation?
        
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
            "ethics_score": 0-10,
            "ethics_justification": "Your detailed justification...",
            "strengths": ["Strength 1", "Strength 2", ...],
            "critiques": ["Critique 1", "Critique 2", ...],
            "improvement_suggestions": ["Suggestion 1", "Suggestion 2", ...],
            "overall_assessment": "Your detailed overall assessment...",
            "recommendation": "accept", "revise", or "reject",
            "literature_references": ["Reference 1", "Reference 2", ...]
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
                data["testability_score"] + 
                data["ethics_score"]
            ) / 5.0
            
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
            
            ## Recommendation
            {data["recommendation"].upper()}
            
            ## Literature References
            {chr(10).join(['- ' + r for r in data.get("literature_references", [])])}
            """
            
            # Create the review
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
                improvement_suggestions=data["improvement_suggestions"]
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