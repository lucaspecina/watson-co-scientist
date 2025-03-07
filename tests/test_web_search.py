"""
Test the web search functionality.
"""

import sys
import os
import asyncio
import json
from dotenv import load_dotenv

# Add src to the path so we can import from it
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Import our modules
from src.tools.web_search import WebSearchTool, ScientificLiteratureSearch
from src.core.models import Citation, Hypothesis, ResearchGoal, HypothesisSource

# Load environment variables
load_dotenv()

async def test_web_search():
    """Test the web search functionality."""
    print("Testing web search functionality...")
    
    # Create a web search tool
    web_search = WebSearchTool(provider="tavily")
    
    # Test basic search
    query = "latest research on CRISPR gene editing"
    print(f"\nPerforming search for: '{query}'")
    
    results = await web_search.search(query, count=3, search_type="scientific")
    
    print(f"Found {len(results)} results:")
    for i, result in enumerate(results):
        print(f"\n{i+1}. {result.get('title', 'Untitled')}")
        print(f"   URL: {result.get('url', 'No URL')}")
        print(f"   Snippet: {result.get('snippet', 'No snippet')[:100]}...")
        
    # Test fetching content
    if results:
        print("\nFetching content for the first result...")
        url = results[0].get("url")
        content = await web_search.fetch_content(url)
        
        if content:
            print(f"Fetched {len(content)} characters of content")
            print(f"Sample: {content[:200]}...")
        else:
            print("Failed to fetch content")
    
    return True

async def test_scientific_literature_search():
    """Test the scientific literature search functionality."""
    print("\nTesting scientific literature search functionality...")
    
    # Create a scientific literature search tool
    literature_search = ScientificLiteratureSearch(provider="tavily")
    
    # Test literature search with citations
    query = "neuroplasticity and aging"
    print(f"\nPerforming scientific literature search for: '{query}'")
    
    result = await literature_search.search_with_citations(query, max_results=3)
    
    # Print search results
    print(f"Found {len(result.get('results', []))} search results")
    
    # Print citations
    citations = result.get("citations", [])
    print(f"Extracted {len(citations)} citations:")
    
    for i, citation in enumerate(citations):
        print(f"\n{i+1}. {citation.get('title', 'Untitled')}")
        print(f"   URL: {citation.get('url', 'No URL')}")
        print(f"   Snippet: {citation.get('snippet', 'No snippet')[:100]}...")
        
    # Create a hypothesis with citations
    if citations:
        print("\nCreating a hypothesis with citations...")
        
        # Convert dictionary citations to Citation objects
        citation_objects = []
        for citation_data in citations:
            citation = Citation(
                title=citation_data.get("title", "Unknown"),
                url=citation_data.get("url", ""),
                authors=[],
                snippet=citation_data.get("snippet", ""),
                source="test_search"
            )
            citation_objects.append(citation)
            
        # Create a research goal
        research_goal = ResearchGoal(
            text="Understand the relationship between neuroplasticity and aging"
        )
        
        # Create a hypothesis with citations
        hypothesis = Hypothesis(
            title="Neuroplasticity may be maintained in aging through specific targeted interventions",
            description="This hypothesis proposes that while neuroplasticity naturally declines with aging, specific cognitive, physical, and social interventions may help maintain or even enhance neuroplastic capacity in older adults.",
            summary="Targeted interventions may maintain neuroplasticity in aging",
            supporting_evidence=["Several studies suggest exercise benefits neuroplasticity", "Cognitive training shows promise for maintaining brain function"],
            citations=citation_objects,
            creator="test",
            source=HypothesisSource.SYSTEM,
            literature_grounded=True,
            metadata={"research_goal_id": research_goal.id}
        )
        
        # Verify the hypothesis has citations
        print(f"Created hypothesis with {len(hypothesis.citations)} citations")
        print(f"Hypothesis title: {hypothesis.title}")
        print(f"First citation: {hypothesis.citations[0].title}")
        
    return True

async def main():
    """Main function to run the tests."""
    # Test web search
    await test_web_search()
    
    # Test scientific literature search
    await test_scientific_literature_search()
    
    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())