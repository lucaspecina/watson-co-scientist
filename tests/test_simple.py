"""
Simple test script to verify core functionality.
"""

import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to the path so we can import from it
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Import our models for testing
from src.core.models import Citation, Hypothesis, ResearchGoal, HypothesisSource

def test_citation_model():
    """Test the Citation model."""
    print("Testing Citation model...")
    
    # Create a sample citation
    citation = Citation(
        title="The role of gut microbiome in Alzheimer's disease",
        authors=["Smith, J", "Johnson, P"],
        year=2023,
        journal="Journal of Neuroscience",
        url="https://example.org/papers/123",
        doi="10.1234/example.5678",
        snippet="This study found significant alterations in gut microbiome composition in Alzheimer's patients.",
        source="tavily_search"
    )
    
    # Verify citation fields
    assert citation.title == "The role of gut microbiome in Alzheimer's disease"
    assert len(citation.authors) == 2
    assert citation.year == 2023
    assert citation.journal == "Journal of Neuroscience"
    
    print(f"Citation created successfully: {citation.title}")
    return citation

def test_hypothesis_with_citations():
    """Test creating a hypothesis with citations."""
    print("\nTesting Hypothesis with citations...")
    
    # Create some citations
    citation1 = Citation(
        title="The role of gut microbiome in Alzheimer's disease",
        authors=["Smith, J", "Johnson, P"],
        year=2023,
        journal="Journal of Neuroscience",
        url="https://example.org/papers/123",
        snippet="This study found significant alterations in gut microbiome composition in Alzheimer's patients."
    )
    
    citation2 = Citation(
        title="Gut-brain axis in Parkinson's disease",
        authors=["Brown, L", "Davis, M"],
        year=2022,
        journal="Neurobiology of Disease",
        url="https://example.org/papers/456",
        snippet="This review discusses the importance of gut-brain signaling in Parkinson's disease pathology."
    )
    
    # Create a research goal
    research_goal = ResearchGoal(
        text="Explore potential relationships between gut microbiome composition and neurodegenerative diseases"
    )
    
    # Create a hypothesis with citations
    hypothesis = Hypothesis(
        title="Gut microbiome metabolites influence neuroinflammation in neurodegenerative diseases",
        description="This hypothesis proposes that specific metabolites produced by gut microbiota can cross the blood-brain barrier and modulate neuroinflammatory processes that contribute to the progression of neurodegenerative diseases.",
        summary="Gut metabolites may influence neuroinflammation in neurodegeneration",
        supporting_evidence=["Altered gut microbiome in Alzheimer's and Parkinson's patients", "Blood-brain barrier permeability changes in neurodegeneration"],
        citations=[citation1, citation2],
        creator="test_script",
        source=HypothesisSource.SYSTEM,
        literature_grounded=True,
        metadata={"research_goal_id": research_goal.id}
    )
    
    # Verify hypothesis fields
    assert hypothesis.title == "Gut microbiome metabolites influence neuroinflammation in neurodegenerative diseases"
    assert hypothesis.literature_grounded == True
    assert len(hypothesis.citations) == 2
    assert hypothesis.citations[0].title == "The role of gut microbiome in Alzheimer's disease"
    assert hypothesis.source == HypothesisSource.SYSTEM
    
    print(f"Hypothesis created successfully: {hypothesis.title}")
    print(f"Number of citations: {len(hypothesis.citations)}")
    print(f"First citation: {hypothesis.citations[0].title}")
    print(f"Literature grounded: {hypothesis.literature_grounded}")
    
    return hypothesis

def main():
    """Main test function."""
    print("Starting simple tests to verify core functionality...")
    
    # Test citation model
    citation = test_citation_model()
    
    # Test hypothesis with citations
    hypothesis = test_hypothesis_with_citations()
    
    print("\nAll tests passed!")

if __name__ == "__main__":
    main()