from agents import Agent, Runner, function_tool
import sys
import os

from pydantic import BaseModel, Field
from typing import Literal, Optional

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.semantic_search import search_product

# Define the input model for query filters
class QueryFilters(BaseModel):
    brand: Optional[Literal["Zara", "Levi's", "H&M", "Uniqlo", "Adidas"]] = Field(None, description="Filter by brand")
    category: Optional[Literal["dresses", "pants", "shirts", "sweaters", "t-shirts", "skirts", "jackets"]] = Field(None, description="Filter by category")
    price_min: Optional[float] = Field(None, description="Minimum price filter")
    price_max: Optional[float] = Field(None, description="Maximum price filter")

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
    
    return search_product(query=query, top_k=top_k, score_threshold=score_threshold, filters=filters_dict)

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
Price range: $12.99 - $149.99

Be conversational, helpful, and focus on understanding what the user really wants to achieve with their clothing purchase.""",
    tools=[search_qdrant],
    tool_use_behavior="run_llm_again"
)

async def run_agent(user_input: str):
    result = await Runner.run(shopping_agent, user_input)
    return result.final_output

if __name__ == "__main__":
    import asyncio
    user_query = input("Enter your search query: ")
    result = asyncio.run(run_agent(user_query))
    print(result)

