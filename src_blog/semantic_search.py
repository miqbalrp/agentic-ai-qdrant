from qdrant_client import QdrantClient, models
import openai

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_client = openai.Client(api_key=openai_api_key)
print("OpenAI client initialized.")

# Initialize Qdrant client
qdrant_url = "http://localhost:6333"
qdrant_client = QdrantClient(url=qdrant_url, timeout=60.0)
print("Qdrant client initialized.")

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
    
    result = models.Filter(must=filter_conditions) if filter_conditions else None
    return result

def search_product(query, top_k=5, score_threshold=0.2, filters=None):
    # Get query embedding
    embedding_model = "text-embedding-3-small"
    response = openai_client.embeddings.create(input=query, model=embedding_model)
    query_vector = response.data[0].embedding
    print(f"Generated query embedding with shape: {len(query_vector)}")

    filter_conditions = None
    if filters:
        filter_conditions = build_filter_conditions(filters)

    # Search in Qdrant collection
    collection_name = "product_catalog"
    results = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        score_threshold=score_threshold,
        with_payload=True,
        query_filter=filter_conditions
    ).points
    print(f"Found {len(results)} results for query '{query}'.")

    # Process results
    processed_results = [
        {
            'score': result.score,
            'name': result.payload['name'],
            'brand': result.payload['brand'],
            'price': result.payload['price'],
            'color': result.payload['color'],
            'size': result.payload['size'],
            'description': result.payload['description'],
            'category': result.payload['category'],
            'material': result.payload['material'],
            'url': result.payload['url']
        }
        for result in results
    ]
    print(f"Processed {len(processed_results)} results.")

    return processed_results

if __name__ == "__main__":
    query = "warm clothes for winter"
    top_k = 5
    score_threshold = 0.2
    filters = {
        'brand': 'H&M',
        'category': 'sweaters',
        'price_min': 10,
        'price_max': 100
    }

    results = search_product(query, top_k, score_threshold, filters)
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['name']} ({result['brand']}) - ${result['price']} - Score: {result['score']:.3f} - Description: {result['description']} - Material: {result['material']} - Color: {result['color']}")
    if not results:
        print("No products matched the search criteria.")