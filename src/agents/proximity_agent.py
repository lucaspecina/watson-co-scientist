"""
Proximity Agent for calculating similarity between hypotheses.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Set, Tuple

from .base_agent import BaseAgent
from ..config.config import SystemConfig
from ..core.models import Hypothesis, ResearchGoal

logger = logging.getLogger("co_scientist")

class ProximityAgent(BaseAgent):
    """
    Agent responsible for calculating similarity between hypotheses.
    """
    
    def __init__(self, config: SystemConfig):
        """
        Initialize the Proximity agent.
        
        Args:
            config (SystemConfig): The system configuration.
        """
        super().__init__("proximity", config)
    
    async def calculate_similarity(self, 
                               hypothesis1: Hypothesis, 
                               hypothesis2: Hypothesis, 
                               research_goal: ResearchGoal) -> float:
        """
        Calculate similarity between two hypotheses.
        
        Args:
            hypothesis1 (Hypothesis): The first hypothesis.
            hypothesis2 (Hypothesis): The second hypothesis.
            research_goal (ResearchGoal): The research goal.
            
        Returns:
            float: Similarity score between 0 and 1.
        """
        logger.info(f"Calculating similarity between hypotheses {hypothesis1.id} and {hypothesis2.id}")
        
        # Build the prompt
        prompt = f"""
        You are evaluating the similarity between two scientific hypotheses for the following research goal:
        
        Research Goal:
        {research_goal.text}
        
        Hypothesis A:
        Title: {hypothesis1.title}
        Summary: {hypothesis1.summary}
        Description: {hypothesis1.description}
        
        Hypothesis B:
        Title: {hypothesis2.title}
        Summary: {hypothesis2.summary}
        Description: {hypothesis2.description}
        
        Compare these hypotheses based on:
        1. Core ideas and mechanisms
        2. Scientific foundations and principles
        3. Proposed methods or approaches
        4. Potential outcomes or implications
        
        Determine a similarity score between 0 and 1, where:
        - 0 = Completely different, addressing different aspects of the research goal
        - 0.25 = Different core ideas but some overlapping elements
        - 0.5 = Moderate similarity with overlapping ideas but significant differences in approach
        - 0.75 = Highly similar with minor differences in implementation or focus
        - 1 = Nearly identical hypotheses
        
        Format your response as a JSON object with the following structure:
        
        ```json
        {{
            "comparison": "Detailed comparison between the hypotheses...",
            "similarities": ["Similarity 1", "Similarity 2", ...],
            "differences": ["Difference 1", "Difference 2", ...],
            "similarity_score": 0.0-1.0
        }}
        ```
        
        Be objective and precise in your evaluation.
        """
        
        # Generate similarity assessment
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
            
            # Get the similarity score
            similarity_score = float(data["similarity_score"])
            
            # Ensure the score is between 0 and 1
            similarity_score = max(0.0, min(1.0, similarity_score))
            
            logger.info(f"Similarity between hypotheses {hypothesis1.id} and {hypothesis2.id}: {similarity_score:.2f}")
            return similarity_score
            
        except Exception as e:
            logger.error(f"Error parsing similarity assessment from response: {e}")
            logger.debug(f"Response: {response}")
            
            # Return a default similarity score in case of error
            return 0.5
    
    async def build_similarity_groups(self, 
                                  hypotheses: List[Hypothesis], 
                                  research_goal: ResearchGoal,
                                  similarity_threshold: float = 0.7,
                                  max_comparisons: Optional[int] = None) -> Dict[str, Set[str]]:
        """
        Build groups of similar hypotheses.
        
        Args:
            hypotheses (List[Hypothesis]): The list of hypotheses.
            research_goal (ResearchGoal): The research goal.
            similarity_threshold (float): The threshold for similarity.
            max_comparisons (Optional[int]): Maximum number of comparisons to perform.
            
        Returns:
            Dict[str, Set[str]]: Dictionary mapping hypothesis IDs to sets of similar hypothesis IDs.
        """
        logger.info(f"Building similarity groups for {len(hypotheses)} hypotheses with threshold {similarity_threshold}")
        
        # Initialize similarity groups
        similarity_groups: Dict[str, Set[str]] = {h.id: set() for h in hypotheses}
        
        # Calculate the number of comparisons to perform
        n = len(hypotheses)
        total_comparisons = (n * (n - 1)) // 2
        
        # Limit the number of comparisons if specified
        if max_comparisons and max_comparisons < total_comparisons:
            logger.info(f"Limiting to {max_comparisons} comparisons out of {total_comparisons} possible")
            # Focus on comparing hypotheses with higher Elo ratings
            sorted_hypotheses = sorted(hypotheses, key=lambda h: h.elo_rating, reverse=True)
            
            # Calculate similarity for top hypotheses
            top_n = int((1 + (1 + 8 * max_comparisons) ** 0.5) / 2)
            top_n = min(top_n, n)
            hypotheses_to_compare = sorted_hypotheses[:top_n]
        else:
            hypotheses_to_compare = hypotheses
        
        # Calculate similarity between each pair of hypotheses
        comparisons_done = 0
        for i, h1 in enumerate(hypotheses_to_compare):
            for h2 in hypotheses_to_compare[i+1:]:
                # Skip if we've reached the maximum number of comparisons
                if max_comparisons and comparisons_done >= max_comparisons:
                    break
                    
                # Calculate similarity
                similarity = await self.calculate_similarity(h1, h2, research_goal)
                comparisons_done += 1
                
                # Add to similarity groups if above threshold
                if similarity >= similarity_threshold:
                    similarity_groups[h1.id].add(h2.id)
                    similarity_groups[h2.id].add(h1.id)
        
        logger.info(f"Completed {comparisons_done} similarity comparisons")
        
        # Log some statistics
        groups_with_similar = sum(1 for group in similarity_groups.values() if group)
        total_similarities = sum(len(group) for group in similarity_groups.values())
        logger.info(f"Found {groups_with_similar} hypotheses with at least one similar hypothesis")
        logger.info(f"Total similarity relationships: {total_similarities // 2}")
        
        return similarity_groups
    
    async def cluster_hypotheses(self,
                            hypotheses: List[Hypothesis],
                            research_goal: ResearchGoal,
                            num_clusters: int = 5) -> Dict[int, List[str]]:
        """
        Cluster hypotheses into groups based on their content.
        
        Args:
            hypotheses (List[Hypothesis]): The list of hypotheses.
            research_goal (ResearchGoal): The research goal.
            num_clusters (int): The number of clusters to create.
            
        Returns:
            Dict[int, List[str]]: Dictionary mapping cluster IDs to lists of hypothesis IDs.
        """
        logger.info(f"Clustering {len(hypotheses)} hypotheses into {num_clusters} clusters")
        
        if len(hypotheses) <= num_clusters:
            # If we have fewer hypotheses than clusters, put each in its own cluster
            return {i: [h.id] for i, h in enumerate(hypotheses)}
        
        # Build the prompt
        hypotheses_text = "\n\n".join([
            f"Hypothesis {i+1} (ID: {h.id}):\nTitle: {h.title}\nSummary: {h.summary}"
            for i, h in enumerate(hypotheses)
        ])
        
        prompt = f"""
        You are analyzing a set of scientific hypotheses for the following research goal:
        
        Research Goal:
        {research_goal.text}
        
        Your task is to cluster the following hypotheses into {num_clusters} groups based on their similarity in terms of core ideas, mechanisms, approaches, and scientific foundations.
        
        Hypotheses:
        {hypotheses_text}
        
        For each cluster:
        1. Assign hypotheses that are conceptually related or address similar aspects of the research goal
        2. Provide a brief label or description for the cluster
        3. Explain why these hypotheses belong together
        
        Format your response as a JSON object with the following structure:
        
        ```json
        {{
            "clusters": [
                {{
                    "cluster_id": 0,
                    "label": "Brief label for cluster 0...",
                    "hypothesis_ids": ["id1", "id2", ...],
                    "rationale": "Explanation of why these hypotheses form a cluster..."
                }},
                ...
            ]
        }}
        ```
        
        Ensure that:
        - Every hypothesis is assigned to exactly one cluster
        - The clusters are meaningfully different from each other
        - The number of clusters is exactly {num_clusters}
        """
        
        # Generate clusters
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
            
            # Convert to the desired format
            clusters = {}
            for cluster in data["clusters"]:
                cluster_id = int(cluster["cluster_id"])
                hypothesis_ids = cluster["hypothesis_ids"]
                clusters[cluster_id] = hypothesis_ids
            
            # Check that all hypotheses are assigned
            assigned_ids = set()
            for ids in clusters.values():
                assigned_ids.update(ids)
                
            if len(assigned_ids) != len(hypotheses):
                logger.warning(f"Not all hypotheses were assigned to clusters ({len(assigned_ids)} out of {len(hypotheses)})")
                
                # Add any missing hypotheses to the smallest cluster
                missing_ids = set(h.id for h in hypotheses) - assigned_ids
                if missing_ids:
                    smallest_cluster = min(clusters.items(), key=lambda item: len(item[1]))
                    clusters[smallest_cluster[0]].extend(list(missing_ids))
            
            logger.info(f"Clustered hypotheses into {len(clusters)} clusters")
            return clusters
            
        except Exception as e:
            logger.error(f"Error parsing clusters from response: {e}")
            logger.debug(f"Response: {response}")
            
            # Create a simple clustering (one hypothesis per cluster) in case of error
            hypotheses_per_cluster = max(1, len(hypotheses) // num_clusters)
            clusters = {}
            for i in range(num_clusters):
                start_idx = i * hypotheses_per_cluster
                end_idx = min(start_idx + hypotheses_per_cluster, len(hypotheses))
                if start_idx >= len(hypotheses):
                    break
                clusters[i] = [h.id for h in hypotheses[start_idx:end_idx]]
                
            return clusters