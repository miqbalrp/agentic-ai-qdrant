from qdrant_client import QdrantClient, models
import openai
import sys
import os

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.config as CONFIG

def build_filter_conditions(filters):
    """Build Qdrant filter conditions from user input."""
    filter_conditions = []
    
    if 'brand' in filters:
        filter_conditions.append(models.FieldCondition(
            key='brand',
            match=models.MatchValue(value=filters['brand'])
        ))
    if 'category' in filters:
        filter_conditions.append(models.FieldCondition(
            key='category',
            match=models.MatchValue(value=filters['category'])
        ))
    if 'price_min' in filters:
        filter_conditions.append(models.FieldCondition(
            key='price',
            range=models.Range(gte=filters['price_min'])
        ))
    if 'price_max' in filters:
        filter_conditions.append(models.FieldCondition(
            key='price',
            range=models.Range(lte=filters['price_max'])
        ))
    
    return models.Filter(must=filter_conditions) if filter_conditions else None

def search_product(query, top_k=5, score_threshold=0.2, filters=None):
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
    
    # Build filter conditions if provided
    filter_conditions = build_filter_conditions(filters) if filters else None
    if filter_conditions:
        print(f"Applying filters: {filter_conditions}")
    else:
        print("No filters applied, searching all products.")

    # Search Qdrant with optional filters
    results = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        score_threshold=score_threshold,
        with_payload=True,
        query_filter=filter_conditions
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
    query = "warm jacket Adidas"
    filters = {
        'brand': 'Adidas',
        'category': 'jackets',
        'price_min': 50,
        'price_max': 150
    }
    results = search_product(query, filters=filters)
    print(f"Found {len(results)} results for '{query}'")
    for result in results:
        print(f"- {result['name']} (${result['price']}) - Score: {result['score']:.3f}")

if __name__ == "__main__":
    main()
