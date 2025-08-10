from agents import Agent, Runner, function_tool
from src.semantic_search import search_product

@function_tool
def search_qdrant(query: str, top_k: int = 5, score_threshold: float = 0.2) -> list:
    """
    Search for clothing products based on a natural language query.
    
    Args:
        query (str): The search query.
        top_k (int): Number of results to return.
        score_threshold (float): Minimum similarity score to include in results.
    
    Returns:
        list: List of matching products with details.
    """
    return search_product(query, top_k, score_threshold)

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

