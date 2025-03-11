"""
Basic tests for the Raul Co-Scientist system.
"""

import os
import asyncio
import unittest
import shutil
from unittest.mock import patch, AsyncMock, MagicMock

from src.core.system import CoScientistSystem
from src.core.models import (
    ResearchGoal, 
    Hypothesis,
    Review,
    TournamentMatch
)
from src.core.llm_provider import LLMProvider

class TestCoScientistSystem(unittest.TestCase):
    """Test the Co-Scientist system."""
    
    @patch('src.core.llm_provider.LLMProvider.create_provider')
    def setUp(self, mock_create_provider):
        """Set up the test environment."""
        # Create a mock provider
        mock_provider = MagicMock()
        mock_provider.generate = AsyncMock(return_value={"content": "Test response", "usage": {"total_tokens": 100}})
        mock_create_provider.return_value = mock_provider
        
        # Create a temporary data directory
        self.test_data_dir = "test_data"
        os.makedirs(self.test_data_dir, exist_ok=True)
        
        # Initialize system with test directory
        self.system = CoScientistSystem(data_dir=self.test_data_dir)
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove test data directory
        if os.path.exists(self.test_data_dir):
            shutil.rmtree(self.test_data_dir)
    
    def test_system_initialization(self):
        """Test system initialization."""
        self.assertIsNotNone(self.system)
        self.assertIsNotNone(self.system.supervisor)
        self.assertIsNotNone(self.system.generation)
        self.assertIsNotNone(self.system.reflection)
        self.assertIsNotNone(self.system.ranking)
        self.assertIsNotNone(self.system.proximity)
        self.assertIsNotNone(self.system.evolution)
        self.assertIsNotNone(self.system.meta_review)
    
    @patch('src.agents.supervisor_agent.SupervisorAgent.parse_research_goal')
    def test_analyze_research_goal(self, mock_parse):
        """Test analyzing a research goal."""
        # Configure the mock to return a dictionary
        plan_config = {
            "main_objective": "Test objective",
            "scope": "Test scope",
            "constraints": ["Constraint 1", "Constraint 2"],
            "preferences": {
                "novelty_required": True,
                "practical_application_focus": True,
                "interdisciplinary_approach": True,
                "time_sensitivity": "medium-term"
            },
            "domains": ["Domain 1", "Domain 2"],
            "evaluation_criteria": ["Criterion 1", "Criterion 2"]
        }
        mock_parse.return_value = plan_config
        
        # Run test
        research_goal = asyncio.run(self.system.analyze_research_goal("Test research goal"))
        
        # Assert results
        self.assertIsNotNone(research_goal)
        self.assertEqual(research_goal.text, "Test research goal")
        self.assertEqual(self.system.current_research_goal, research_goal)
        
        # Assert state initialization
        self.assertEqual(self.system.current_state["num_hypotheses"], 0)
        self.assertEqual(self.system.current_state["num_reviews"], 0)
        self.assertEqual(self.system.current_state["iterations_completed"], 0)

if __name__ == '__main__':
    unittest.main()