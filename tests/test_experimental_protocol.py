"""
Test the experimental protocol generation functionality.
"""

import asyncio
import logging
import os
import sys
import unittest
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.config import load_config
from src.core.models import ResearchGoal
from src.agents.generation_agent import GenerationAgent
from src.agents.reflection_agent import ReflectionAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestExperimentalProtocol(unittest.TestCase):
    """Test the experimental protocol generation and review functionality."""
    
    def setUp(self):
        """Set up the test."""
        self.config = load_config()
        self.generation_agent = GenerationAgent(self.config)
        self.reflection_agent = ReflectionAgent(self.config)
        
        # Create a sample research goal
        self.research_goal = ResearchGoal(
            text="Develop a novel approach to increase the efficiency of CRISPR-Cas9 gene editing in plant cells for improved crop traits.",
            preferences={"novelty_required": True, "practical_application_focus": True},
            constraints={"time_sensitivity": "medium-term"}
        )
        
    def test_protocol_generation(self):
        """Test generating an experimental protocol for a hypothesis."""
        async def run_test():
            # First, generate a hypothesis
            hypotheses = await self.generation_agent.generate_initial_hypotheses(
                self.research_goal, num_hypotheses=1
            )
            self.assertEqual(len(hypotheses), 1)
            hypothesis = hypotheses[0]
            
            # Now generate a protocol for this hypothesis
            protocol = await self.generation_agent.generate_experimental_protocol(
                hypothesis, self.research_goal
            )
            
            # Verify the protocol
            self.assertIsNotNone(protocol)
            self.assertEqual(protocol.hypothesis_id, hypothesis.id)
            self.assertIsNotNone(protocol.title)
            self.assertIsNotNone(protocol.description)
            self.assertGreater(len(protocol.steps), 0)
            self.assertGreater(len(protocol.materials), 0)
            self.assertGreater(len(protocol.equipment), 0)
            self.assertIsNotNone(protocol.expected_results)
            
            # Now review the protocol
            review = await self.reflection_agent.review_protocol(
                protocol, hypothesis, self.research_goal
            )
            
            # Verify the review
            self.assertIsNotNone(review)
            self.assertEqual(review.hypothesis_id, hypothesis.id)
            self.assertEqual(review.metadata.get("protocol_id"), protocol.id)
            self.assertIsNotNone(review.overall_score)
            self.assertGreater(len(review.strengths), 0)
            
            logger.info(f"Generated hypothesis: {hypothesis.title}")
            logger.info(f"Generated protocol: {protocol.title}")
            logger.info(f"Protocol review score: {review.overall_score}")
            
            return hypothesis, protocol, review
            
        # Run the async test
        hypothesis, protocol, review = asyncio.run(run_test())
        
        # Print the protocol details for manual inspection
        print("\n==== Generated Hypothesis ====")
        print(f"Title: {hypothesis.title}")
        print(f"Summary: {hypothesis.summary}")
        
        print("\n==== Generated Protocol ====")
        print(f"Title: {protocol.title}")
        print(f"Description: {protocol.description}")
        print("\nSteps:")
        for i, step in enumerate(protocol.steps, 1):
            print(f"{i}. {step}")
        
        print("\nMaterials:")
        for material in protocol.materials:
            print(f"- {material}")
            
        print("\nEquipment:")
        for equipment in protocol.equipment:
            print(f"- {equipment}")
            
        print(f"\nExpected Results: {protocol.expected_results}")
        
        print("\n==== Protocol Review ====")
        print(f"Overall Score: {review.overall_score}")
        print("\nStrengths:")
        for strength in review.strengths:
            print(f"- {strength}")
            
        print("\nAreas for Improvement:")
        for critique in review.critiques:
            print(f"- {critique}")

if __name__ == "__main__":
    unittest.main()