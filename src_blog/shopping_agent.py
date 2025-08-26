from agents import Agent, Runner, function_tool
from pydantic import BaseModel, Field
from typing import Literal, Optional

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.semantic_search import search_product

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define the input model for query filters
class QueryFilters(BaseModel):
    brand: Optional[Literal["Zara", "Levi's", "H&M", "Uniqlo", "Adidas"]] = Field(None, description="Filter by brand")
    category: Optional[Literal["dresses", "pants", "shirts", "sweaters", "t-shirts", "skirts", "jackets"]] = Field(None, description="Filter by category")
    price_min: Optional[float] = Field(None, description="Minimum price filter")
    price_max: Optional[float] = Field(None, description="Maximum price filter")

# Search function for Qdrant
@function_tool
def search_qdrant(query: str, filters: QueryFilters = QueryFilters(), top_k: int = 5, score_threshold: float = 0.2) -> list:
    """
    Search for clothing products based on a natural language query.
    
    Args:
        query (str): The search query.
        filters (QueryFilters): Optional filters for brand, category, price range, etc.
        top_k (int): Number of results to return.
        score_threshold (float): Minimum similarity score to include in results.
    Returns:
        list: List of matching products with details.
    """
    
    # Convert QueryFilters to dictionary, excluding None values
    filters_dict = filters.model_dump(exclude_none=True)
    
    try:
        results = search_product(query=query, top_k=top_k, score_threshold=score_threshold, filters=filters_dict)
        return results
    except Exception as e:
        raise

# Define the shopping agent
shopping_agent = Agent(
    name="Shopping Agent",
    instructions="""You are an expert shopping assistant specializing in clothing and fashion. Your role is to help users find the perfect clothing items based on their needs and preferences.

When helping users:
1. Ask clarifying questions if their request is vague (e.g., occasion, size, budget, style preferences)
2. Use the search_qdrant tool to find relevant products based on their query
3. Present results in a friendly, organized manner with key details like price, brand, material, and colors
4. Provide styling suggestions or alternatives when appropriate
5. Help users compare different options based on their criteria

Available product categories: dresses, pants, shirts, sweaters, t-shirts, skirts, jackets
Available brands: Zara, Levi's, H&M, Uniqlo, Adidas

Be conversational, helpful, and focus on understanding what the user really wants to achieve with their clothing purchase.""",
    tools=[search_qdrant],
    tool_use_behavior="run_llm_again"
)

# Run the agent with user input
async def run_agent(user_input: str):
    try:
        result = await Runner.run(shopping_agent, user_input)
        return result.final_output
    except Exception as e:
        raise

# Main entry point for running the agent
# This allows the agent to be run directly from the command line
if __name__ == "__main__":
    import asyncio
    user_query = input("Enter your search query: ")
    
    try:
        result = asyncio.run(run_agent(user_query))
        print(result)
    except Exception as e:
        print(f"Error: {str(e)}")