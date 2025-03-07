"""
Test the web search functionality using mock responses.
"""

import sys
import os
import asyncio
import json
from unittest.mock import patch

# Add src to the path so we can import from it
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Import our modules
from src.tools.web_search import WebSearchTool, ScientificLiteratureSearch
from src.core.models import Citation, Hypothesis, ResearchGoal, HypothesisSource

# Mock data for Tavily API responses
MOCK_TAVILY_RESPONSE = {
    "answer": "Recent CRISPR research focuses on improving precision, developing new delivery methods, and expanding applications to complex diseases. Key advances include base editing, prime editing, and CRISPR-Cas systems beyond Cas9.",
    "results": [
        {
            "title": "Recent advances in CRISPR gene editing technology",
            "url": "https://example.com/article1",
            "content": "This review discusses recent advances in CRISPR gene editing including base editing, prime editing, and new Cas variants.",
            "domain": "example.com",
            "published_date": "2023-05-15"
        },
        {
            "title": "CRISPR applications in cancer research",
            "url": "https://example.com/article2",
            "content": "Researchers are using CRISPR to identify cancer driver genes and develop potential therapeutic approaches.",
            "domain": "example.com",
            "published_date": "2023-06-20"
        }
    ]
}

async def test_mock_web_search():
    """Test the web search functionality with mock data."""
    print("Testing web search functionality with mock data...")
    
    # Create a web search tool
    web_search = WebSearchTool(provider="tavily", api_key="mock_key")
    
    # Mock the search method
    with patch.object(web_search, '_tavily_search', return_value=asyncio.Future()) as mock_search:
        mock_search.return_value.set_result([
            {
                "title": "Recent advances in CRISPR gene editing technology",
                "url": "https://example.com/article1",
                "snippet": "This review discusses recent advances in CRISPR gene editing including base editing, prime editing, and new Cas variants.",
                "source": "tavily",
                "publication_date": "2023-05-15"
            },
            {
                "title": "CRISPR applications in cancer research",
                "url": "https://example.com/article2",
                "snippet": "Researchers are using CRISPR to identify cancer driver genes and develop potential therapeutic approaches.",
                "source": "tavily",
                "publication_date": "2023-06-20"
            }
        ])
        
        # Test basic search
        query = "latest research on CRISPR gene editing"
        print(f"\nPerforming mock search for: '{query}'")
        
        results = await web_search.search(query, count=3)
        
        print(f"Found {len(results)} results:")
        for i, result in enumerate(results):
            print(f"\n{i+1}. {result.get('title', 'Untitled')}")
            print(f"   URL: {result.get('url', 'No URL')}")
            print(f"   Snippet: {result.get('snippet', 'No snippet')}")
            
    # Mock the fetch content method
    with patch.object(web_search, 'fetch_content', return_value=asyncio.Future()) as mock_fetch:
        mock_fetch.return_value.set_result("This is mocked content for testing purposes.")
        
        # Mock the search_and_fetch method to use our mocked methods
        with patch.object(web_search, 'search', return_value=asyncio.Future()) as mock_search_fetch:
            mock_search_fetch.return_value.set_result([
                {
                    "title": "Recent advances in CRISPR gene editing technology",
                    "url": "https://example.com/article1",
                    "snippet": "This review discusses recent advances in CRISPR gene editing including base editing, prime editing, and new Cas variants.",
                    "source": "tavily",
                    "publication_date": "2023-05-15"
                }
            ])
            
            print("\nTesting search_and_fetch with mock data...")
            results = await web_search.search_and_fetch(query, max_results=1)
            
            print(f"Found {len(results)} results with content:")
            for result in results:
                print(f"Title: {result.get('title')}")
                print(f"Content preview: {result.get('content')[:50]}...")
    
    return True

async def test_mock_scientific_literature_search():
    """Test the scientific literature search functionality with mock data."""
    print("\nTesting scientific literature search functionality with mock data...")
    
    # Create a scientific literature search tool
    literature_search = ScientificLiteratureSearch(provider="tavily", api_key="mock_key")
    
    # Mock the search_with_citations method
    with patch.object(literature_search.web_search, 'search_and_fetch', return_value=asyncio.Future()) as mock_search:
        mock_search.return_value.set_result([
            {
                "title": "Neuroplasticity in aging: recent advances and future directions",
                "url": "https://example.com/neuro1",
                "snippet": "This review discusses recent findings in neuroplasticity research related to aging populations.",
                "content": "Extended content about neuroplasticity and aging research findings...",
                "source": "tavily",
                "publication_date": "2023-04-10"
            },
            {
                "title": "Cognitive training interventions to enhance neuroplasticity in older adults",
                "url": "https://example.com/neuro2",
                "snippet": "This study explores how cognitive training can promote neuroplasticity in aging brains.",
                "content": "Extended content about cognitive training interventions...",
                "source": "tavily",
                "publication_date": "2023-03-15"
            }
        ])
        
        # Test literature search
        query = "neuroplasticity and aging"
        print(f"\nPerforming mock scientific literature search for: '{query}'")
        
        result = await literature_search.search_with_citations(query, max_results=2)
        
        # Print search results
        print(f"Found {len(result.get('results', []))} search results")
        
        # Print citations
        citations = result.get("citations", [])
        print(f"Extracted {len(citations)} citations:")
        
        for i, citation in enumerate(citations):
            print(f"\n{i+1}. {citation.get('title', 'Untitled')}")
            print(f"   URL: {citation.get('url', 'No URL')}")
            print(f"   Snippet: {citation.get('snippet', '')[:50]}...")
        
    return True

async def test_mock_hypothesis_with_citations():
    """Test creating a hypothesis with citations using mock data."""
    print("\nTesting hypothesis creation with citations...")
    
    # Create mock citations
    citations = [
        Citation(
            title="Neuroplasticity in aging: recent advances and future directions",
            url="https://example.com/neuro1",
            authors=["Smith, J", "Johnson, A"],
            year=2023,
            journal="Journal of Neuroscience",
            snippet="This review discusses recent findings in neuroplasticity research related to aging populations."
        ),
        Citation(
            title="Cognitive training interventions to enhance neuroplasticity in older adults",
            url="https://example.com/neuro2",
            authors=["Brown, R", "Davis, M"],
            year=2023,
            journal="Aging Research",
            snippet="This study explores how cognitive training can promote neuroplasticity in aging brains."
        )
    ]
    
    # Create a research goal
    research_goal = ResearchGoal(
        text="Understand the relationship between neuroplasticity and aging"
    )
    
    # Create a hypothesis with citations
    hypothesis = Hypothesis(
        title="Targeted cognitive interventions may enhance neuroplasticity in aging populations",
        description="This hypothesis proposes that specific, personalized cognitive interventions designed to target individual cognitive weaknesses may significantly enhance neuroplasticity in aging populations, leading to improved cognitive outcomes and potentially slowing cognitive decline.",
        summary="Personalized cognitive interventions may enhance neuroplasticity in aging",
        supporting_evidence=["Prior studies show cognitive training benefits", "Animal models demonstrate neurogenesis in response to environmental enrichment"],
        citations=citations,
        creator="test",
        source=HypothesisSource.SYSTEM,
        literature_grounded=True,
        metadata={"research_goal_id": research_goal.id}
    )
    
    # Verify the hypothesis has citations
    print(f"Created hypothesis with {len(hypothesis.citations)} citations")
    print(f"Hypothesis title: {hypothesis.title}")
    print(f"First citation: {hypothesis.citations[0].title}")
    print(f"Literature grounded: {hypothesis.literature_grounded}")
    
    return hypothesis

async def main():
    """Main function to run the tests."""
    # Test web search with mock data
    await test_mock_web_search()
    
    # Test scientific literature search with mock data
    await test_mock_scientific_literature_search()
    
    # Test hypothesis with citations
    await test_mock_hypothesis_with_citations()
    
    print("\nAll mock tests completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())