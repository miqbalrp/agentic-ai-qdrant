from qdrant_client import QdrantClient
import openai
import src.config as CONFIG

def search_product(query, top_k=5, score_threshold=0.2):
    """Complete search workflow: embed query, search Qdrant, return results."""
    
    qdrant_url = CONFIG.QDRANT_URL
    openai_api_key = CONFIG.OPENAI_API_KEY
    collection_name = CONFIG.QDRANT_COLLECTION_NAME
    embedding_model = CONFIG.EMBEDDING_MODEL

    # Initialize clients
    qdrant_client = QdrantClient(url=qdrant_url)
    openai_client = openai.Client(api_key=openai_api_key)

    # Get query embedding
    response = openai_client.embeddings.create(input=query, model=embedding_model)
    query_vector = response.data[0].embedding
    
    # Search Qdrant
    results = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        score_threshold=score_threshold
    ).points
    
    # Return structured data for AI agent
    if not results:
        return []
    
    return [
        {
            'score': result.score,
            'name': result.payload['name'],
            'brand': result.payload['brand'],
            'price': result.payload['price'],
            'color': result.payload['color'],
            'size': result.payload['size'],
            'description': result.payload['description'],
            'category': result.payload['category'],
            'material': result.payload['material']
        }
        for result in results
    ]

def main():
    """Simple test interface."""
    query = "warm jacket"
    results = search_product(query)
    print(f"Found {len(results)} results for '{query}'")
    for result in results:
        print(f"- {result['name']} (${result['price']}) - Score: {result['score']:.3f}")

if __name__ == "__main__":
    main()
