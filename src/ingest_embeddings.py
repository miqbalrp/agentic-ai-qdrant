from qdrant_client import QdrantClient, models
import pandas as pd
import numpy as np
from uuid import uuid4
import logging
import os
import sys
from datetime import datetime

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.config as CONFIG

qdrant_url = CONFIG.QDRANT_URL
qdrant_collection_name = CONFIG.QDRANT_COLLECTION_NAME
openai_api_key = CONFIG.OPENAI_API_KEY
embedding_model = CONFIG.EMBEDDING_MODEL
embed_products_log_file = CONFIG.EMBED_PRODUCTS_LOG_FILE
dataset_path = CONFIG.DATASET_PATH
embedding_file_path = CONFIG.EMBEDDING_FILE

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/ingest_embeddings.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info("Starting embedding ingestion process")
logger.info(f"Connecting to Qdrant at: {qdrant_url}")

try:
    client = QdrantClient(url=qdrant_url, timeout=60.0)
    logger.info("Successfully connected to Qdrant client")
except Exception as e:
    logger.error(f"Failed to connect to Qdrant: {str(e)}")
    sys.exit(1)

# Load dataset and embeddings
logger.info(f"Loading dataset from: {dataset_path}")
logger.info(f"Loading embeddings from: {embedding_file_path}")

try:
    df = pd.read_json(dataset_path)
    logger.info(f"Successfully loaded {len(df)} products from dataset")

    vectors = np.load(embedding_file_path)
    logger.info(f"Successfully loaded embeddings with shape: {vectors.shape}")
    
except Exception as e:
    logger.error(f"Failed to load data: {str(e)}")
    sys.exit(1)

vector_dimension = vectors.shape[1]
logger.info(f"Target collection: {qdrant_collection_name}")
logger.info(f"Vector dimension: {vector_dimension}")

# Create collection (delete existing if present)
try:
    if client.collection_exists(qdrant_collection_name):
        logger.warning(f"Collection '{qdrant_collection_name}' already exists. Deleting...")
        client.delete_collection(qdrant_collection_name)
        logger.info(f"Collection '{qdrant_collection_name}' deleted successfully")

    logger.info(f"Creating new collection '{qdrant_collection_name}' with {vector_dimension} dimensions...")
    client.create_collection(
        collection_name=qdrant_collection_name,
        vectors_config=models.VectorParams(size=vector_dimension, distance=models.Distance.COSINE),
    )
    logger.info(f"Collection '{qdrant_collection_name}' created successfully")
    
except Exception as e:
    logger.error(f"Failed to create collection: {str(e)}")
    sys.exit(1)  

# Prepare points for insertion
logger.info(f"Preparing {len(df)} points for insertion")
points = []

try:
    start_time = datetime.now()
    
    for idx, (_, row) in enumerate(df.iterrows()):
        point = models.PointStruct(
            id=str(uuid4()),  # Generate unique ID
            vector=vectors[idx].tolist(),
            payload={
                "product_id": row["id"],
                "name": row["name"],
                "category": row["category"],
                "brand": row["brand"],
                "price": row["price"],
                "color": row["color"],
                "material": row["material"],
                "size": row["size"],
                "description": row["description"]
            }
        )
        points.append(point)
        
        # Log progress every 25 items
        if (idx + 1) % 25 == 0:
            logger.debug(f"Prepared {idx + 1}/{len(df)} points")
    
    preparation_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"Successfully prepared {len(points)} points in {preparation_time:.2f} seconds")
    
except Exception as e:
    logger.error(f"Failed to prepare points: {str(e)}")
    sys.exit(1)

# Insert points into Qdrant
logger.info(f"Starting insertion of {len(points)} points into Qdrant...")

try:
    start_time = datetime.now()
    
    client.upsert(
        collection_name=qdrant_collection_name,
        points=points,
        wait=True  # Wait for the operation to complete
    )
    
    end_time = datetime.now()
    insertion_time = (end_time - start_time).total_seconds()
    
    # Verify insertion by checking collection info
    collection_info = client.get_collection(qdrant_collection_name)
    points_count = collection_info.points_count
    
    logger.info(f"Successfully inserted points into Qdrant")
    logger.info(f"Insertion took {insertion_time:.2f} seconds")
    logger.info(f"Collection now contains {points_count} points")
            
except Exception as e:
    logger.error(f"Failed to insert points into Qdrant: {str(e)}")
    sys.exit(1)

# Final summary
logger.info("=" * 50)
logger.info("EMBEDDING INGESTION SUMMARY")
logger.info("=" * 50)
logger.info(f"Dataset: {dataset_path}")
logger.info(f"Embeddings: {embedding_file_path}")
logger.info(f"Collection: {qdrant_collection_name}")
logger.info(f"Points ingested: {len(points)}")
logger.info(f"Vector dimension: {vector_dimension}")
logger.info(f"Total ingestion time: {insertion_time:.2f} seconds")
logger.info("Embedding ingestion completed successfully!")
logger.info("=" * 50)