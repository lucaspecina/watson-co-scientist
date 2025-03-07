"""
Test the Tavily search API directly using their Python client.
"""

import os
from dotenv import load_dotenv
from tavily import TavilyClient

# Load environment variables
load_dotenv()

def test_tavily_search():
    """Test the Tavily search API directly."""
    
    # Get the API key from environment variables
    api_key = os.environ.get("TAVILY_API_KEY")
    
    if not api_key:
        print("No Tavily API key found in environment variables")
        return False
        
    print(f"Found Tavily API key: {api_key[:5]}...{api_key[-5:]}")
    
    # Create the Tavily client
    client = TavilyClient(api_key=api_key)
    
    # Perform a search
    query = "latest research on gut microbiome and neurodegenerative diseases"
    
    try:
        print(f"Performing search for: '{query}'")
        response = client.search(query=query, search_depth="advanced", include_answer=True, search_type="scientific")
        
        # Print the response
        if "answer" in response:
            print(f"\nAnswer generated:\n{response['answer'][:200]}...")
            
        # Print the results
        if "results" in response:
            print(f"\nFound {len(response['results'])} results:")
            
            for i, result in enumerate(response["results"][:3]):
                print(f"\n{i+1}. {result.get('title', 'Untitled')}")
                print(f"   URL: {result.get('url', 'No URL')}")
                print(f"   Content: {result.get('content', 'No content')[:100]}...")
                
        return True
        
    except Exception as e:
        print(f"Error using Tavily API: {e}")
        return False
        
if __name__ == "__main__":
    test_tavily_search()