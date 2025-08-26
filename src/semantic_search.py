from qdrant_client import QdrantClient, models
import openai
import sys
import os
import logging
from datetime import datetime

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/semantic_search.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.config as CONFIG

def build_filter_conditions(filters):
    """Build Qdrant filter conditions from user input."""
    logger.debug(f"Building filter conditions from: {filters}")
    filter_conditions = []
    
    if 'brand' in filters:
        filter_conditions.append(models.FieldCondition(
            key='brand',
            match=models.MatchValue(value=filters['brand'])
        ))
        logger.debug(f"Added brand filter: {filters['brand']}")
        
    if 'category' in filters:
        filter_conditions.append(models.FieldCondition(
            key='category',
            match=models.MatchValue(value=filters['category'])
        ))
        logger.debug(f"Added category filter: {filters['category']}")
        
    if 'price_min' in filters:
        filter_conditions.append(models.FieldCondition(
            key='price',
            range=models.Range(gte=filters['price_min'])
        ))
        logger.debug(f"Added minimum price filter: {filters['price_min']}")
        
    if 'price_max' in filters:
        filter_conditions.append(models.FieldCondition(
            key='price',
            range=models.Range(lte=filters['price_max'])
        ))
        logger.debug(f"Added maximum price filter: {filters['price_max']}")
    
    result = models.Filter(must=filter_conditions) if filter_conditions else None
    logger.debug(f"Built filter with {len(filter_conditions)} conditions")
    return result

def search_product(query, top_k=5, score_threshold=0.2, filters=None):
    """Complete search workflow: embed query, search Qdrant, return results."""
    
    logger.info(f"Starting product search for query: '{query}'")
    logger.info(f"Search parameters - top_k: {top_k}, score_threshold: {score_threshold}")
    
    qdrant_url = CONFIG.QDRANT_URL
    openai_api_key = CONFIG.OPENAI_API_KEY
    collection_name = CONFIG.QDRANT_COLLECTION_NAME
    embedding_model = CONFIG.EMBEDDING_MODEL

    # Initialize clients
    logger.debug(f"Initializing clients - Qdrant: {qdrant_url}, Model: {embedding_model}")
    try:
        qdrant_client = QdrantClient(url=qdrant_url)
        openai_client = openai.Client(api_key=openai_api_key)
        logger.debug("Successfully initialized Qdrant and OpenAI clients")
    except Exception as e:
        logger.error(f"Failed to initialize clients: {str(e)}")
        raise

    # Get query embedding
    logger.info(f"Generating embedding for query using model: {embedding_model}")
    try:
        response = openai_client.embeddings.create(input=query, model=embedding_model)
        query_vector = response.data[0].embedding
        logger.debug(f"Embedding dimension: {len(query_vector)}")
    except Exception as e:
        logger.error(f"Failed to generate embedding: {str(e)}")
        raise
    
    # Build filter conditions if provided
    filter_conditions = None
    if filters:
        logger.info(f"Applying filters: {filters}")
        filter_conditions = build_filter_conditions(filters)
    else:
        logger.info("No filters applied, searching all products")

    # Search Qdrant with optional filters
    logger.info(f"Searching collection '{collection_name}'")
    try:
        results = qdrant_client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
            query_filter=filter_conditions
        ).points
        
        logger.info(f"Search completed, found {len(results)} results")
        
        if results:
            logger.debug(f"Top result score: {results[0].score:.4f}")
            logger.debug(f"Lowest result score: {results[-1].score:.4f}")
        
    except Exception as e:
        logger.error(f"Failed to search Qdrant: {str(e)}")
        raise
    
    # Return structured data for AI agent
    if not results:
        logger.warning("No results found matching the criteria")
        return []
    
    logger.info(f"Processing {len(results)} results for return")
    try:
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
        
        logger.info(f"Successfully processed {len(processed_results)} results")
        logger.debug(f"Sample result: {processed_results[0]['name'] if processed_results else 'None'}")
        
        return processed_results
        
    except Exception as e:
        logger.error(f"Failed to process search results: {str(e)}")
        raise

def main():
    """Test interface with comprehensive logging."""
    logger.info("=" * 50)
    logger.info("SEMANTIC SEARCH TEST")
    logger.info("=" * 50)
    
    query = "warm jacket Adidas"
    filters = {
        'brand': 'Adidas',
        'category': 'jackets',
        'price_min': 50,
        'price_max': 150
    }
    
    logger.info(f"Test query: '{query}'")
    logger.info(f"Test filters: {filters}")
    
    try:
        start_time = datetime.now()
        results = search_product(query, filters=filters)
        total_time = (datetime.now() - start_time).total_seconds()
        
        logger.info("=" * 30)
        logger.info("SEARCH RESULTS")
        logger.info("=" * 30)
        logger.info(f"Found {len(results)} results for '{query}'")
        logger.info(f"Total search time: {total_time:.3f} seconds")
        
        for i, result in enumerate(results, 1):
            logger.info(f"{i}. {result['name']} ({result['brand']}) - ${result['price']} - Score: {result['score']:.3f}")
            
        if not results:
            logger.warning("No products matched the search criteria")
        
        logger.info("=" * 30)
        logger.info("TEST COMPLETED SUCCESSFULLY")
        logger.info("=" * 30)
        
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
