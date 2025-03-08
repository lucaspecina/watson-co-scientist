"""
Evolution Agent for improving existing hypotheses.
"""

import json
import logging
import random
import os
from typing import Dict, List, Any, Optional, Tuple, Set

from .base_agent import BaseAgent
from ..config.config import SystemConfig
from ..core.models import (
    Hypothesis, 
    ResearchGoal, 
    Review, 
    HypothesisSource, 
    HypothesisStatus, 
    ResearchFocus,
    Citation
)
from ..tools.web_search import WebSearchTool
from ..tools.domain_specific.knowledge_manager import DomainKnowledgeManager
from ..tools.domain_specific.ontology import DomainOntology

logger = logging.getLogger("co_scientist")

class EvolutionAgent(BaseAgent):
    """
    Agent responsible for improving existing hypotheses.
    
    The Evolution Agent employs various strategies to improve hypotheses, including:
    1. Standard improvements based on reviews
    2. Combinations of multiple promising hypotheses
    3. Cross-domain inspiration for novel perspectives
    4. Domain-specific knowledge integration for scientific grounding
    5. Out-of-the-box creative approaches
    6. Simplification for clarity and testability
    7. Targeted evolution based on research focus areas
    8. Analogical reasoning from other scientific fields
    """
    
    def __init__(self, config: SystemConfig):
        """
        Initialize the Evolution agent.
        
        Args:
            config (SystemConfig): The system configuration.
        """
        super().__init__("evolution", config)
        
        # Initialize web search tool if enabled
        self.web_search = None
        if config.web_search_enabled:
            self.web_search = WebSearchTool(config.web_search_api_key)
            
        # Initialize domain knowledge manager
        self.domain_knowledge = DomainKnowledgeManager(config.get("domain_knowledge", {}))
        
        # Track available ontologies
        self.ontologies = {}
        self._load_ontologies()
        
        # Evolution strategies with weights (for random selection)
        self.evolution_strategies = {
            "improve": 0.3,           # Standard improvements
            "combine": 0.2,           # Combine multiple hypotheses
            "domain_knowledge": 0.2,  # Integrate domain-specific knowledge
            "out_of_box": 0.1,        # Generate creative, unconventional hypotheses
            "cross_domain": 0.1,      # Apply concepts from other domains
            "simplify": 0.1,          # Simplify complex hypotheses
        }
        
    def _load_ontologies(self):
        """Load available domain ontologies."""
        try:
            # Load ontologies from the ontologies directory
            ontology_dir = os.path.join(os.path.dirname(__file__), "../tools/domain_specific/ontologies")
            if os.path.exists(ontology_dir):
                for filename in os.listdir(ontology_dir):
                    if filename.endswith("_ontology.json"):
                        domain = filename.replace("_ontology.json", "")
                        self.ontologies[domain] = DomainOntology(domain)
                        logger.info(f"Registered {domain} ontology for Evolution Agent")
        except Exception as e:
            logger.warning(f"Error loading ontologies: {e}")
    
    async def select_evolution_strategy(self, 
                                 hypothesis: Hypothesis,
                                 research_goal: ResearchGoal,
                                 reviews: List[Review] = None) -> str:
        """
        Select an appropriate evolution strategy based on hypothesis characteristics and reviews.
        
        Args:
            hypothesis (Hypothesis): The hypothesis to evolve
            research_goal (ResearchGoal): The research goal
            reviews (List[Review], optional): Reviews of the hypothesis. Defaults to None.
            
        Returns:
            str: The selected strategy
        """
        # If no reviews, default to standard improvement
        if not reviews:
            return "improve"
            
        # Extract review critiques and scores
        critiques = []
        for review in reviews:
            critiques.extend(review.critiques)
            
        avg_novelty = sum(r.novelty_score for r in reviews if r.novelty_score is not None) / len([r for r in reviews if r.novelty_score is not None]) if any(r.novelty_score is not None for r in reviews) else None
        avg_testability = sum(r.testability_score for r in reviews if r.testability_score is not None) / len([r for r in reviews if r.testability_score is not None]) if any(r.testability_score is not None for r in reviews) else None
            
        # Analyze critiques to determine best strategy
        critique_text = " ".join(critiques).lower()
        
        # Check for signs that domain knowledge integration would help
        domain_knowledge_indicators = ["lacks scientific grounding", "needs more evidence", "insufficient literature support",
                                      "lacks specificity", "should cite", "references needed", "theory gap"]
        
        # Check for signs that simplification would help
        simplification_indicators = ["too complex", "hard to understand", "needs clarity", "convoluted", 
                                    "overly complicated", "difficult to test", "too many components"]
        
        # Check for signs that out-of-box thinking would help
        out_of_box_indicators = ["conventional", "lacks novelty", "similar to existing", "not original", 
                                "incremental", "derivative", "standard approach"]
        
        # Count indicators in critiques
        domain_count = sum(1 for indicator in domain_knowledge_indicators if indicator in critique_text)
        simplify_count = sum(1 for indicator in simplification_indicators if indicator in critique_text)
        creative_count = sum(1 for indicator in out_of_box_indicators if indicator in critique_text)
        
        # Make decision based on counts and scores
        if creative_count > 0 or (avg_novelty is not None and avg_novelty < 5.0):
            if creative_count > domain_count and creative_count > simplify_count:
                # Novelty is the biggest issue - try creative approaches
                return random.choice(["out_of_box", "cross_domain"])
                
        if domain_count > 0:
            if domain_count > simplify_count:
                # Domain knowledge is the biggest issue
                return "domain_knowledge"
                
        if simplify_count > 0 or (avg_testability is not None and avg_testability < 5.0):
            if simplify_count > domain_count:
                # Complexity is the biggest issue
                return "simplify"
                
        # Default to standard improvement
        return "improve"
    
    async def evolve_hypothesis(self, 
                           hypothesis: Hypothesis, 
                           research_goal: ResearchGoal,
                           reviews: List[Review] = None) -> Hypothesis:
        """
        Evolve a hypothesis using the most appropriate strategy.
        
        Args:
            hypothesis (Hypothesis): The hypothesis to evolve
            research_goal (ResearchGoal): The research goal
            reviews (List[Review], optional): Reviews of the hypothesis. Defaults to None.
            
        Returns:
            Hypothesis: The evolved hypothesis
        """
        # Select an evolution strategy
        strategy = await self.select_evolution_strategy(hypothesis, research_goal, reviews)
        logger.info(f"Selected evolution strategy: {strategy} for hypothesis {hypothesis.id}")
        
        # Apply the selected strategy
        if strategy == "improve":
            return await self.improve_hypothesis(hypothesis, research_goal, reviews)
        elif strategy == "domain_knowledge":
            return await self.improve_with_domain_knowledge(hypothesis, research_goal, reviews)
        elif strategy == "out_of_box":
            return await self.generate_out_of_box_hypothesis(research_goal, [hypothesis])
        elif strategy == "cross_domain":
            return await self.apply_cross_domain_inspiration(hypothesis, research_goal, reviews)
        elif strategy == "simplify":
            return await self.simplify_hypothesis(hypothesis, research_goal)
        else:
            # Fallback to standard improvement
            return await self.improve_hypothesis(hypothesis, research_goal, reviews)
    
    async def improve_hypothesis(self, 
                            hypothesis: Hypothesis, 
                            research_goal: ResearchGoal,
                            reviews: List[Review] = None) -> Hypothesis:
        """
        Improve an existing hypothesis based on reviews and research goal.
        
        Args:
            hypothesis (Hypothesis): The hypothesis to improve.
            research_goal (ResearchGoal): The research goal.
            reviews (List[Review]): List of reviews for the hypothesis.
            
        Returns:
            Hypothesis: The improved hypothesis.
        """
        logger.info(f"Improving hypothesis {hypothesis.id}: {hypothesis.title}")
        
        # Prepare review context if provided
        review_context = ""
        if reviews:
            review_text = "\n\n".join([
                f"Review {i+1}:\n{review.text}"
                for i, review in enumerate(reviews)
            ])
            
            review_context = f"""
            Previous Reviews:
            {review_text}
            """
        
        # Check for active research focus areas
        focus_areas = []
        try:
            # Access system data safely through class-level attributes
            from ..core.system import CoScientistSystem
            
            # Find the system instance
            system = None
            # If this module has a global 'system' variable
            import sys
            main_module = sys.modules.get('__main__')
            if main_module and hasattr(main_module, 'system') and isinstance(main_module.system, CoScientistSystem):
                system = main_module.system
            
            # Get database access
            db = system.db if system else None
            
            if db:
                focus_areas = db.get_active_research_focus(research_goal.id)
        except Exception as e:
            logger.warning(f"Error fetching research focus areas: {e}")
            
        # Prepare focus area context
        focus_context = ""
        if focus_areas:
            focus_text = "\n\n".join([
                f"Focus Area {i+1}:\nTitle: {focus.title}\nDescription: {focus.description}\nKeywords: {', '.join(focus.keywords)}\nPriority: {focus.priority}"
                for i, focus in enumerate(focus_areas)
            ])
            
            focus_context = f"""
            Active Research Focus Areas:
            {focus_text}
            
            IMPORTANT: Consider these research focus areas in your improvements, especially those with higher priority.
            """

        # Build the prompt
        prompt = f"""
        You are improving an existing scientific hypothesis based on the research goal, previous reviews, and active research focus areas.
        
        Research Goal:
        {research_goal.text}
        
        Original Hypothesis:
        Title: {hypothesis.title}
        Summary: {hypothesis.summary}
        Description: {hypothesis.description}
        Supporting Evidence: {', '.join(hypothesis.supporting_evidence)}
        
        {review_context}
        
        {focus_context}
        
        Your task is to create an improved version of this hypothesis that:
        1. Addresses any weaknesses or critiques from the reviews
        2. Maintains or enhances the strengths identified in reviews
        3. Makes the hypothesis more precise, testable, and aligned with the research goal
        4. Incorporates additional relevant scientific principles or evidence
        5. Improves the clarity and coherence of the hypothesis
        6. When possible, aligns with or addresses the active research focus areas
        
        Do not simply tweak the hypothesis; make substantive improvements while preserving the core idea.
        
        Format your response as a JSON object with the following structure:
        
        ```json
        {{
            "title": "New concise title for the hypothesis",
            "summary": "Brief summary (1-2 sentences)",
            "description": "Detailed description of the improved hypothesis (1-2 paragraphs)",
            "supporting_evidence": ["Evidence 1", "Evidence 2", ...],
            "improvements_made": ["Improvement 1", "Improvement 2", ...],
            "rationale": "Explanation of why these improvements address the critiques and strengthen the hypothesis"
        }}
        ```
        """
        
        # Generate improved hypothesis
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
            
            # Create the improved hypothesis
            improved = Hypothesis(
                title=data["title"],
                description=data["description"],
                summary=data["summary"],
                supporting_evidence=data["supporting_evidence"],
                creator="evolution",
                source=HypothesisSource.EVOLVED,
                parent_hypotheses=[hypothesis.id],
                metadata={
                    "research_goal_id": research_goal.id,
                    "improvements_made": data.get("improvements_made", []),
                    "improvement_rationale": data.get("rationale", ""),
                    "evolution_strategy": "improve"
                }
            )
            
            logger.info(f"Created improved hypothesis: {improved.title}")
            return improved
            
        except Exception as e:
            logger.error(f"Error parsing improved hypothesis from response: {e}")
            logger.debug(f"Response: {response}")
            
            # Return the original hypothesis in case of error
            return hypothesis
    
    async def improve_with_domain_knowledge(self,
                                       hypothesis: Hypothesis,
                                       research_goal: ResearchGoal,
                                       reviews: List[Review] = None) -> Hypothesis:
        """
        Improve a hypothesis by integrating domain-specific knowledge.
        
        Args:
            hypothesis (Hypothesis): The hypothesis to improve
            research_goal (ResearchGoal): The research goal
            reviews (List[Review], optional): Reviews of the hypothesis. Defaults to None.
            
        Returns:
            Hypothesis: The improved hypothesis with domain knowledge
        """
        logger.info(f"Improving hypothesis {hypothesis.id} with domain knowledge")
        
        # Detect potential domains from research goal and hypothesis
        domains = await self._detect_domains(research_goal, hypothesis)
        
        # Prepare domain knowledge context
        domain_context = ""
        citations = []
        
        # If we have domain ontologies, use them
        ontology_context = ""
        for domain in domains:
            if domain in self.ontologies:
                ontology = self.ontologies[domain]
                if ontology.initialize():
                    # Find relevant concepts based on hypothesis and research goal
                    search_terms = self._extract_key_terms(hypothesis, research_goal)
                    
                    relevant_concepts = []
                    for term in search_terms:
                        concepts = ontology.search_concepts(term, limit=3)
                        for concept in concepts:
                            # Add concept if it's not already in the list
                            if not any(c["id"] == concept["id"] for c in relevant_concepts):
                                relevant_concepts.append(concept)
                    
                    # Format concepts for the prompt
                    if relevant_concepts:
                        concepts_text = "\n\n".join([
                            f"Concept: {concept['name']}\n"
                            f"Description: {concept['description']}\n"
                            f"Related concepts: {', '.join([rel['name'] for rel in ontology.get_related_concepts(concept['id'])[:5]])}"
                            for concept in relevant_concepts[:5]  # Limit to top 5
                        ])
                        
                        ontology_context += f"""
                        {domain.capitalize()} Domain Concepts:
                        {concepts_text}
                        """
        
        # Search for domain-specific literature if domain knowledge manager is available
        literature_context = ""
        try:
            # Make sure domain knowledge manager is initialized
            await self.domain_knowledge.initialize(domains)
            
            # Generate search query based on hypothesis
            query = f"{hypothesis.title} {' '.join(self._extract_key_terms(hypothesis, research_goal))}"
            
            # Search literature
            results = await self.domain_knowledge.search(query, domains=domains, limit=3)
            
            # Format results for the prompt
            for domain, domain_results in results.items():
                if domain_results:
                    results_text = "\n\n".join([
                        f"Title: {result['title']}\n"
                        f"Authors: {', '.join(result['authors']) if 'authors' in result and result['authors'] else 'Unknown'}\n"
                        f"Summary: {result.get('abstract', result.get('snippet', ''))[:300]}..."
                        for result in domain_results[:3]  # Limit to top 3
                    ])
                    
                    literature_context += f"""
                    {domain.capitalize()} Literature:
                    {results_text}
                    """
                    
                    # Create citations for the results
                    for result in domain_results:
                        citation = Citation(
                            title=result["title"],
                            authors=result.get("authors", []),
                            year=result.get("year", ""),
                            journal=result.get("journal", ""),
                            url=result.get("url", ""),
                            snippet=result.get("abstract", result.get("snippet", ""))[:200],
                            source=result.get("source", "unknown")
                        )
                        citations.append(citation)
        except Exception as e:
            logger.warning(f"Error searching domain literature: {e}")
        
        # Combine domain contexts
        if ontology_context or literature_context:
            domain_context = f"""
            Domain-Specific Knowledge:
            {ontology_context}
            {literature_context}
            
            IMPORTANT: Integrate this domain knowledge into the hypothesis to make it more scientifically grounded.
            Use concepts, terminology, and findings from the literature to strengthen the hypothesis.
            """
        
        # Prepare review context if provided
        review_context = ""
        if reviews:
            review_text = "\n\n".join([
                f"Review {i+1}:\n{review.text}"
                for i, review in enumerate(reviews)
            ])
            
            review_context = f"""
            Previous Reviews:
            {review_text}
            """
        
        # Build the prompt
        prompt = f"""
        You are improving a scientific hypothesis by integrating domain-specific knowledge and addressing previous reviews.
        
        Research Goal:
        {research_goal.text}
        
        Original Hypothesis:
        Title: {hypothesis.title}
        Summary: {hypothesis.summary}
        Description: {hypothesis.description}
        Supporting Evidence: {', '.join(hypothesis.supporting_evidence)}
        
        {review_context}
        
        {domain_context}
        
        Your task is to create an improved version of this hypothesis that:
        1. Incorporates relevant domain-specific concepts, terminology, and findings
        2. Addresses any weaknesses identified in the reviews
        3. Is more scientifically precise and grounded in the literature
        4. Makes testable predictions based on established domain knowledge
        5. Connects to existing scientific understanding in the field
        6. Uses appropriate scientific terminology from the domain
        
        Format your response as a JSON object with the following structure:
        
        ```json
        {{
            "title": "New scientifically precise title for the hypothesis",
            "summary": "Brief summary grounded in domain knowledge (1-2 sentences)",
            "description": "Detailed description with domain-specific concepts (1-2 paragraphs)",
            "supporting_evidence": ["Evidence 1", "Evidence 2", ...],
            "domain_concepts_used": ["Concept 1", "Concept 2", ...],
            "scientific_improvements": ["Improvement 1", "Improvement 2", ...],
            "testable_predictions": ["Prediction 1", "Prediction 2", ...]
        }}
        ```
        """
        
        # Generate improved hypothesis
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
            
            # Create the improved hypothesis
            improved = Hypothesis(
                title=data["title"],
                description=data["description"],
                summary=data["summary"],
                supporting_evidence=data["supporting_evidence"],
                creator="evolution_domain",
                source=HypothesisSource.EVOLVED,
                parent_hypotheses=[hypothesis.id],
                citations=[c.id for c in citations],  # Add citations from domain literature
                literature_grounded=True,
                tags={"domain_enhanced"},
                metadata={
                    "research_goal_id": research_goal.id,
                    "domains": domains,
                    "domain_concepts_used": data.get("domain_concepts_used", []),
                    "scientific_improvements": data.get("scientific_improvements", []),
                    "testable_predictions": data.get("testable_predictions", []),
                    "evolution_strategy": "domain_knowledge"
                }
            )
            
            logger.info(f"Created domain-enhanced hypothesis: {improved.title}")
            return improved
            
        except Exception as e:
            logger.error(f"Error parsing domain-enhanced hypothesis from response: {e}")
            logger.debug(f"Response: {response}")
            
            # Fallback to standard improvement
            return await self.improve_hypothesis(hypothesis, research_goal, reviews)
    
    async def apply_cross_domain_inspiration(self,
                                       hypothesis: Hypothesis,
                                       research_goal: ResearchGoal,
                                       reviews: List[Review] = None) -> Hypothesis:
        """
        Improve a hypothesis by applying concepts and principles from other scientific domains.
        
        Uses the cross-domain synthesizer to gather information from multiple scientific domains
        and apply interdisciplinary concepts to the hypothesis.
        
        Args:
            hypothesis (Hypothesis): The hypothesis to improve
            research_goal (ResearchGoal): The research goal
            reviews (List[Review], optional): Reviews of the hypothesis. Defaults to None.
            
        Returns:
            Hypothesis: The improved hypothesis with cross-domain inspiration
        """
        logger.info(f"Applying cross-domain inspiration to hypothesis {hypothesis.id}")
        
        # Detect primary domain
        primary_domains = await self._detect_domains(research_goal, hypothesis)
        primary_domain = primary_domains[0] if primary_domains else "general"
        
        # Build context from reviews
        review_context = ""
        if reviews:
            review_text = "\n\n".join([
                f"Review {i+1}:\n{review.text}"
                for i, review in enumerate(reviews)
            ])
            
            review_context = f"""
            Previous Reviews:
            {review_text}
            """
        
        # Try to use the cross-domain synthesizer if available
        cross_domain_context = ""
        try:
            # Access system to get domain knowledge manager
            from ..core.system import CoScientistSystem
            from ..tools.domain_specific.cross_domain_synthesizer import CrossDomainSynthesizer
            
            # Find the system instance
            system = None
            import sys
            main_module = sys.modules.get('__main__')
            if main_module and hasattr(main_module, 'system') and isinstance(main_module.system, CoScientistSystem):
                system = main_module.system
            
            if system and hasattr(system, 'domain_knowledge'):
                # Create synthesizer with the domain knowledge manager
                synthesizer = CrossDomainSynthesizer(system.domain_knowledge)
                
                # Create an enhanced query combining research goal and hypothesis
                query = f"{research_goal.text} {hypothesis.title} {' '.join(primary_domains)}"
                
                # Determine potential cross-domains
                cross_domains = {}
                if primary_domain == "biomedicine":
                    cross_domains = ["physics", "computer_science", "chemistry"]
                elif primary_domain == "physics":
                    cross_domains = ["biology", "computer_science", "mathematics"]
                elif primary_domain == "chemistry":
                    cross_domains = ["physics", "biology", "computer_science"]
                elif primary_domain == "computer_science":
                    cross_domains = ["neuroscience", "mathematics", "physics"]
                elif primary_domain == "biology":
                    cross_domains = ["computer_science", "physics", "chemistry"]
                else:
                    cross_domains = ["physics", "biology", "computer_science", "mathematics"]
                
                # Get cross-domain document context - use original domains plus detected ones
                cross_domain_docs = await synthesizer.get_cross_domain_document_context(
                    query, max_documents_per_domain=2
                )
                
                # Format cross-domain information as context
                if cross_domain_docs:
                    cross_domain_sections = []
                    for domain, content in cross_domain_docs.items():
                        if domain != primary_domain:  # Only include cross-domains
                            cross_domain_sections.append(f"--- {domain.upper()} DOMAIN KNOWLEDGE ---\n{content}")
                    
                    if cross_domain_sections:
                        cross_domain_context = "Cross-Domain Knowledge:\n" + "\n\n".join(cross_domain_sections)
                        
                        # If cross-domain context is too large, summarize it
                        if len(cross_domain_context) > 6000:
                            logger.info("Cross-domain context is large, summarizing")
                            
                            summarization_prompt = f"""
                            Summarize the following cross-domain scientific information to identify key concepts, principles,
                            and methodologies that could be applied to improve a hypothesis in {primary_domain} domain.
                            Focus on novel connections and interdisciplinary ideas.
                            
                            {cross_domain_context}
                            """
                            
                            # Summarize the cross-domain context
                            summary = await self.generate(summarization_prompt)
                            cross_domain_context = f"Cross-Domain Knowledge Summary:\n{summary}"
                            
                # If we couldn't get cross-domain context, fall back to listing domains
                if not cross_domain_context:
                    cross_domain_context = f"Consider applying concepts and principles from these domains: {', '.join(cross_domains)}"
                
        except Exception as e:
            logger.warning(f"Error using cross-domain synthesizer: {e}")
            # Define fallback cross-domain areas for inspiration
            cross_domains = {
                "biomedicine": ["physics", "computer_science", "ecology", "mathematics"],
                "physics": ["biology", "computer_science", "economics", "mathematics"],
                "chemistry": ["physics", "biology", "materials_science", "computer_science"],
                "computer_science": ["neuroscience", "economics", "linguistics", "physics"],
                "ecology": ["network_theory", "economics", "physics", "social_science"],
                "general": ["physics", "biology", "computer_science", "mathematics", "economics"]
            }
            
            # Select domains for cross-pollination
            inspiration_domains = cross_domains.get(primary_domain, cross_domains["general"])
            cross_domain_context = f"Inspiration Domains:\n{', '.join(inspiration_domains)}"
        
        # Build the prompt
        prompt = f"""
        You are applying cross-domain inspiration to improve a scientific hypothesis by drawing on concepts, principles, and methodologies from other scientific fields.
        
        Research Goal:
        {research_goal.text}
        
        Original Hypothesis:
        Title: {hypothesis.title}
        Summary: {hypothesis.summary}
        Description: {hypothesis.description}
        Supporting Evidence: {', '.join(hypothesis.supporting_evidence)}
        
        Primary Domain: {primary_domain}
        
        {review_context}
        
        {cross_domain_context}
        
        Your task is to create an improved version of this hypothesis that:
        1. Applies concepts, analogies, models, or principles from other scientific domains to the primary domain
        2. Creates novel connections between the primary domain and other scientific fields
        3. Uses cross-domain thinking to address limitations identified in the reviews
        4. Maintains scientific rigor while introducing innovative perspectives
        5. Provides a fresh viewpoint that could lead to breakthrough insights
        6. Uses the cross-domain knowledge provided to ground your improvements in actual scientific concepts
        
        Format your response as a JSON object with the following structure:
        
        ```json
        {{
            "title": "Cross-domain inspired title for the hypothesis",
            "summary": "Brief summary with cross-domain concepts (1-2 sentences)",
            "description": "Detailed description applying cross-domain thinking (1-2 paragraphs)",
            "supporting_evidence": ["Evidence 1", "Evidence 2", ...],
            "inspiration_domains_used": ["Domain 1", "Domain 2", ...],
            "cross_domain_concepts": ["Concept 1 from Domain X", "Concept 2 from Domain Y", ...],
            "novel_connections": ["Connection 1", "Connection 2", ...],
            "potential_breakthroughs": ["Potential breakthrough 1", "Potential breakthrough 2", ...]
        }}
        ```
        """
        
        # Generate improved hypothesis
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
            
            # Create the improved hypothesis
            improved = Hypothesis(
                title=data["title"],
                description=data["description"],
                summary=data["summary"],
                supporting_evidence=data["supporting_evidence"],
                creator="evolution_cross_domain",
                source=HypothesisSource.EVOLVED,
                parent_hypotheses=[hypothesis.id],
                tags={"cross_domain", "innovative"},
                metadata={
                    "research_goal_id": research_goal.id,
                    "primary_domain": primary_domain,
                    "inspiration_domains": data.get("inspiration_domains_used", []),
                    "cross_domain_concepts": data.get("cross_domain_concepts", []),
                    "novel_connections": data.get("novel_connections", []),
                    "potential_breakthroughs": data.get("potential_breakthroughs", []),
                    "evolution_strategy": "cross_domain"
                }
            )
            
            logger.info(f"Created cross-domain inspired hypothesis: {improved.title}")
            return improved
            
        except Exception as e:
            logger.error(f"Error parsing cross-domain hypothesis from response: {e}")
            logger.debug(f"Response: {response}")
            
            # Fallback to standard improvement
            return await self.improve_hypothesis(hypothesis, research_goal, reviews)
    
    async def _detect_domains(self, research_goal: ResearchGoal, hypothesis: Optional[Hypothesis] = None) -> List[str]:
        """
        Detect relevant scientific domains based on research goal and hypothesis.
        
        Args:
            research_goal (ResearchGoal): The research goal
            hypothesis (Optional[Hypothesis], optional): The hypothesis. Defaults to None.
            
        Returns:
            List[str]: List of detected domains
        """
        # Define known domains and their keywords
        domain_keywords = {
            "biomedicine": ["disease", "drug", "treatment", "medicine", "patient", "clinical", "gene", "protein", 
                           "biology", "cell", "molecular", "cancer", "therapy", "biomarker", "mutation", "tissue", 
                           "receptor", "signaling", "pathology", "virus", "immune", "brain", "neuron"],
            "chemistry": ["chemical", "molecule", "compound", "reaction", "synthesis", "polymer", "catalyst", 
                         "solvent", "acid", "base", "organic", "inorganic", "spectroscopy", "bond", "element", 
                         "structure", "material", "crystal", "solution", "ion", "electrochemical"],
            "physics": ["quantum", "particle", "energy", "force", "field", "relativity", "gravity", "mechanics", 
                       "thermodynamics", "electromagnetism", "nuclear", "optics", "motion", "wave", "radiation", 
                       "quantum", "material", "state", "velocity", "acceleration", "mass", "light"],
            "computer_science": ["algorithm", "data", "software", "programming", "network", "learning", "ai", 
                                "artificial intelligence", "computation", "system", "neural", "machine", "database",
                                "optimization", "computational", "security", "information", "robot", "automation"],
            "ecology": ["ecosystem", "species", "environment", "climate", "conservation", "biodiversity", "habitat", 
                       "population", "evolution", "organism", "sustainability", "pollution", "resource", "forest", 
                       "ocean", "wildlife", "carbon", "nitrogen", "food web", "adaptation", "vegetation"]
        }
        
        # Combine text from research goal and hypothesis
        combined_text = research_goal.text.lower()
        if hypothesis:
            combined_text += " " + hypothesis.title.lower()
            combined_text += " " + hypothesis.description.lower()
            combined_text += " " + hypothesis.summary.lower()
            combined_text += " " + " ".join(hypothesis.supporting_evidence).lower()
        
        # Count domain keywords in the text
        domain_counts = {}
        for domain, keywords in domain_keywords.items():
            count = sum(1 for keyword in keywords if keyword.lower() in combined_text)
            domain_counts[domain] = count
        
        # Sort domains by count
        sorted_domains = sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Return top domains (those with at least one keyword match)
        return [domain for domain, count in sorted_domains if count > 0]
    
    def _extract_key_terms(self, hypothesis: Hypothesis, research_goal: ResearchGoal) -> List[str]:
        """
        Extract key scientific terms from a hypothesis and research goal.
        
        Args:
            hypothesis (Hypothesis): The hypothesis
            research_goal (ResearchGoal): The research goal
            
        Returns:
            List[str]: List of key terms
        """
        # Combine text from hypothesis and research goal
        combined_text = (
            hypothesis.title + " " + 
            hypothesis.summary + " " + 
            research_goal.text
        )
        
        # Extract all words
        words = combined_text.replace(",", " ").replace(".", " ").replace(";", " ").split()
        
        # Keep only likely scientific terms (longer words, not common stopwords)
        stopwords = {"the", "a", "an", "in", "on", "at", "by", "for", "with", "about",
                    "to", "from", "of", "and", "or", "but", "is", "are", "was", "were",
                    "be", "been", "being", "have", "has", "had", "do", "does", "did",
                    "will", "would", "should", "could", "may", "might", "must", "can",
                    "this", "that", "these", "those", "their", "there", "it", "its"}
        
        terms = [word for word in words if len(word) > 4 and word.lower() not in stopwords]
        
        # Prioritize multi-word scientific terms by checking for combinations
        multi_word_terms = []
        for i in range(len(words) - 1):
            if len(words[i]) > 3 and len(words[i+1]) > 3:  # Both words reasonably long
                if words[i].lower() not in stopwords and words[i+1].lower() not in stopwords:
                    multi_word_terms.append(f"{words[i]} {words[i+1]}")
        
        # Return unique terms
        all_terms = list(set(terms + multi_word_terms))
        
        # Sort by length (longer terms often more specific)
        all_terms.sort(key=len, reverse=True)
        
        return all_terms[:10]  # Return top 10 terms
    
    async def combine_hypotheses(self, 
                            hypotheses: List[Hypothesis], 
                            research_goal: ResearchGoal) -> Hypothesis:
        """
        Combine multiple hypotheses into a new, improved hypothesis.
        
        Args:
            hypotheses (List[Hypothesis]): The hypotheses to combine.
            research_goal (ResearchGoal): The research goal.
            
        Returns:
            Hypothesis: The combined hypothesis.
        """
        logger.info(f"Combining {len(hypotheses)} hypotheses")
        
        # Ensure we have at least two hypotheses to combine
        if len(hypotheses) < 2:
            logger.warning("Need at least two hypotheses to combine")
            return hypotheses[0] if hypotheses else None
        
        # Build the prompt
        hypotheses_text = "\n\n".join([
            f"Hypothesis {i+1}:\nTitle: {h.title}\nSummary: {h.summary}\nDescription: {h.description}\nSupporting Evidence: {', '.join(h.supporting_evidence)}"
            for i, h in enumerate(hypotheses)
        ])
        
        prompt = f"""
        You are combining multiple scientific hypotheses into a new, stronger hypothesis that addresses the research goal.
        
        Research Goal:
        {research_goal.text}
        
        Existing Hypotheses:
        {hypotheses_text}
        
        Your task is to create a new hypothesis that:
        1. Combines the strongest elements from each input hypothesis
        2. Resolves any contradictions between the input hypotheses
        3. Creates a synthesis that is more powerful than any individual hypothesis
        4. Is novel, coherent, and directly addresses the research goal
        5. Is scientifically sound and testable
        
        This is not simply a summary; it should be a novel synthesis that represents a conceptual advancement.
        
        Format your response as a JSON object with the following structure:
        
        ```json
        {{
            "title": "Concise title for the combined hypothesis",
            "summary": "Brief summary (1-2 sentences)",
            "description": "Detailed description of the combined hypothesis (1-2 paragraphs)",
            "supporting_evidence": ["Evidence 1", "Evidence 2", ...],
            "elements_used": [
                {{
                    "hypothesis": 1,
                    "elements": ["Element 1 from hypothesis 1", "Element 2 from hypothesis 1", ...]
                }},
                ...
            ],
            "synergy": "Explanation of how the combination creates value beyond the individual hypotheses"
        }}
        ```
        """
        
        # Generate combined hypothesis
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
            
            # Create the combined hypothesis
            combined = Hypothesis(
                title=data["title"],
                description=data["description"],
                summary=data["summary"],
                supporting_evidence=data["supporting_evidence"],
                creator="evolution_combine",
                source=HypothesisSource.COMBINED,
                parent_hypotheses=[h.id for h in hypotheses],
                metadata={
                    "research_goal_id": research_goal.id,
                    "elements_used": data.get("elements_used", []),
                    "synergy": data.get("synergy", "")
                }
            )
            
            logger.info(f"Created combined hypothesis: {combined.title}")
            return combined
            
        except Exception as e:
            logger.error(f"Error parsing combined hypothesis from response: {e}")
            logger.debug(f"Response: {response}")
            
            # Return the highest-rated hypothesis in case of error
            return max(hypotheses, key=lambda h: h.elo_rating)
    
    async def generate_out_of_box_hypothesis(self, 
                                        research_goal: ResearchGoal,
                                        existing_hypotheses: List[Hypothesis] = None) -> Hypothesis:
        """
        Generate an out-of-the-box hypothesis that takes a different approach.
        
        Args:
            research_goal (ResearchGoal): The research goal.
            existing_hypotheses (List[Hypothesis]): Existing hypotheses to diverge from.
            
        Returns:
            Hypothesis: The out-of-the-box hypothesis.
        """
        logger.info(f"Generating out-of-the-box hypothesis for research goal {research_goal.id}")
        
        # Prepare context of existing hypotheses if provided
        existing_context = ""
        if existing_hypotheses:
            top_hypotheses = sorted(existing_hypotheses, key=lambda h: h.elo_rating, reverse=True)[:3]
            
            hypotheses_text = "\n\n".join([
                f"Hypothesis {i+1}:\nTitle: {h.title}\nSummary: {h.summary}"
                for i, h in enumerate(top_hypotheses)
            ])
            
            existing_context = f"""
            Current Top Hypotheses (to diverge from):
            {hypotheses_text}
            """
        
        # Build the prompt
        prompt = f"""
        You are generating an out-of-the-box, creative scientific hypothesis for the following research goal:
        
        Research Goal:
        {research_goal.text}
        
        {existing_context}
        
        Your task is to create a novel hypothesis that:
        1. Takes a radically different approach from conventional thinking and existing hypotheses
        2. Challenges fundamental assumptions in the field
        3. Draws inspiration from other scientific disciplines or paradigm shifts
        4. Is still scientifically plausible and testable
        5. Directly addresses the research goal
        
        The hypothesis should be creative and unconventional but still scientifically sound - not pseudoscience.
        
        Format your response as a JSON object with the following structure:
        
        ```json
        {{
            "title": "Concise title for the out-of-the-box hypothesis",
            "summary": "Brief summary (1-2 sentences)",
            "description": "Detailed description of the hypothesis (1-2 paragraphs)",
            "supporting_evidence": ["Evidence 1", "Evidence 2", ...],
            "unconventional_aspects": ["Aspect 1", "Aspect 2", ...],
            "inspiration": "What inspired this unconventional approach"
        }}
        ```
        """
        
        # Generate out-of-the-box hypothesis
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
            
            # Create the out-of-the-box hypothesis
            hypothesis = Hypothesis(
                title=data["title"],
                description=data["description"],
                summary=data["summary"],
                supporting_evidence=data["supporting_evidence"],
                creator="evolution_out_of_box",
                source=HypothesisSource.EVOLVED,
                tags={"out_of_box", "creative"},
                metadata={
                    "research_goal_id": research_goal.id,
                    "unconventional_aspects": data.get("unconventional_aspects", []),
                    "inspiration": data.get("inspiration", "")
                }
            )
            
            logger.info(f"Created out-of-the-box hypothesis: {hypothesis.title}")
            return hypothesis
            
        except Exception as e:
            logger.error(f"Error parsing out-of-the-box hypothesis from response: {e}")
            logger.debug(f"Response: {response}")
            
            # Create a basic hypothesis in case of error
            hypothesis = Hypothesis(
                title=f"Creative approach to {research_goal.text[:50]}...",
                description="Error parsing generated hypothesis.",
                summary="Error parsing generated hypothesis.",
                supporting_evidence=["Error during generation"],
                creator="evolution_out_of_box",
                source=HypothesisSource.EVOLVED,
                tags={"out_of_box", "error"},
                metadata={
                    "research_goal_id": research_goal.id,
                    "generation_error": str(e)
                }
            )
            
            return hypothesis
    
    async def simplify_hypothesis(self, 
                             hypothesis: Hypothesis, 
                             research_goal: ResearchGoal) -> Hypothesis:
        """
        Simplify a complex hypothesis while preserving its core idea.
        
        Args:
            hypothesis (Hypothesis): The hypothesis to simplify.
            research_goal (ResearchGoal): The research goal.
            
        Returns:
            Hypothesis: The simplified hypothesis.
        """
        logger.info(f"Simplifying hypothesis {hypothesis.id}: {hypothesis.title}")
        
        # Build the prompt
        prompt = f"""
        You are simplifying a complex scientific hypothesis while preserving its core idea and scientific validity.
        
        Research Goal:
        {research_goal.text}
        
        Original Hypothesis:
        Title: {hypothesis.title}
        Summary: {hypothesis.summary}
        Description: {hypothesis.description}
        Supporting Evidence: {', '.join(hypothesis.supporting_evidence)}
        
        Your task is to create a simplified version of this hypothesis that:
        1. Preserves the essential scientific idea and mechanism
        2. Reduces unnecessary complexity and jargon
        3. Makes the hypothesis more accessible and testable
        4. Maintains scientific rigor and alignment with the research goal
        5. Is clearer and more concise
        
        This is not about dumbing down the science, but rather about expressing it more elegantly and clearly.
        
        Format your response as a JSON object with the following structure:
        
        ```json
        {{
            "title": "Simplified title for the hypothesis",
            "summary": "Brief simplified summary (1-2 sentences)",
            "description": "Simplified description of the hypothesis (1-2 paragraphs)",
            "supporting_evidence": ["Evidence 1", "Evidence 2", ...],
            "simplifications_made": ["Simplification 1", "Simplification 2", ...],
            "preserved_elements": ["Essential element 1 preserved", "Essential element 2 preserved", ...]
        }}
        ```
        """
        
        # Generate simplified hypothesis
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
            
            # Create the simplified hypothesis
            simplified = Hypothesis(
                title=data["title"],
                description=data["description"],
                summary=data["summary"],
                supporting_evidence=data["supporting_evidence"],
                creator="evolution_simplify",
                source=HypothesisSource.EVOLVED,
                parent_hypotheses=[hypothesis.id],
                tags={"simplified"},
                metadata={
                    "research_goal_id": research_goal.id,
                    "simplifications_made": data.get("simplifications_made", []),
                    "preserved_elements": data.get("preserved_elements", [])
                }
            )
            
            logger.info(f"Created simplified hypothesis: {simplified.title}")
            return simplified
            
        except Exception as e:
            logger.error(f"Error parsing simplified hypothesis from response: {e}")
            logger.debug(f"Response: {response}")
            
            # Return the original hypothesis in case of error
            return hypothesis