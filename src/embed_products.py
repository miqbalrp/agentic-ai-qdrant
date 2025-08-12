import pandas as pd
import numpy as np
import openai
import sys
import os
import logging
from datetime import datetime

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.config as CONFIG

openai_api_key = CONFIG.OPENAI_API_KEY
embedding_model = CONFIG.EMBEDDING_MODEL
embed_products_log_file = CONFIG.EMBED_PRODUCTS_LOG_FILE
dataset_path = CONFIG.DATASET_PATH
embedding_file_path = CONFIG.EMBEDDING_FILE

# Create logs directory if it doesn't exist (BEFORE setting up logging)
os.makedirs('logs', exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(embed_products_log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load the dataset
logger.info("Starting embedding generation process")
logger.info(f"Loading dataset from: {dataset_path}")

try:
    df = pd.read_json(dataset_path)
    logger.info(f"Successfully loaded {len(df)} products from dataset")
    logger.debug(f"Dataset columns: {list(df.columns)}")
except Exception as e:
    logger.error(f"Failed to load dataset: {str(e)}")
    sys.exit(1)

# Initialize OpenAI client with API key from environment
logger.info("Initializing OpenAI client")
try:
    openai_client = openai.Client(api_key=openai_api_key)
    logger.info("OpenAI client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {str(e)}")
    sys.exit(1)

# Prepare text descriptions
logger.info("Preparing text descriptions for embedding generation")
try:
    texts = [f"{row.name}. {row.description} Material: {row.material}. Color: {row.color}." for row in df.itertuples()]
    logger.info(f"Prepared {len(texts)} text descriptions")
    logger.debug(f"Sample text: {texts[0][:100]}...")
except Exception as e:
    logger.error(f"Failed to prepare text descriptions: {str(e)}")
    sys.exit(1)

# Generate embeddings using OpenAI
logger.info(f"Starting embedding generation using model: {embedding_model}")
logger.info(f"Processing {len(texts)} texts in batch")

try:
    start_time = datetime.now()
    response = openai_client.embeddings.create(input=texts, model=embedding_model)
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    logger.info(f"Successfully generated {len(response.data)} embeddings")
    logger.info(f"Embedding dimension: {len(response.data[0].embedding)}")
    logger.info(f"Generation took {duration:.2f} seconds")
    logger.info(f"Average time per embedding: {duration/len(response.data):.3f} seconds")
    
except Exception as e:
    logger.error(f"Failed to generate embeddings: {str(e)}")
    sys.exit(1)

# Extract only the embedding vectors (not the full response objects)
logger.info("Extracting embedding vectors from API response")
try:
    embeddings = [item.embedding for item in response.data]
    vectors = np.array(embeddings)
    logger.info(f"Created numpy array with shape: {vectors.shape}")
    logger.info(f"Array memory usage: {vectors.nbytes / (1024*1024):.2f} MB")
except Exception as e:
    logger.error(f"Failed to extract embeddings: {str(e)}")
    sys.exit(1)

# Save embeddings using config path
logger.info(f"Saving embeddings to: {embedding_file_path}")

try:
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(embedding_file_path), exist_ok=True)
    
    # Save the embeddings
    start_time = datetime.now()
    np.save(embedding_file_path, vectors, allow_pickle=False)
    end_time = datetime.now()
    save_duration = (end_time - start_time).total_seconds()
    
    # Verify file was created and get size
    file_size = os.path.getsize(embedding_file_path) / (1024*1024)  # MB
    
    logger.info(f"Successfully saved embeddings to disk")
    logger.info(f"File size: {file_size:.2f} MB")
    logger.info(f"Save operation took {save_duration:.3f} seconds")
    
except Exception as e:
    logger.error(f"Failed to save embeddings: {str(e)}")
    sys.exit(1)

# Final summary
logger.info("=" * 50)
logger.info("EMBEDDING GENERATION SUMMARY")
logger.info("=" * 50)
logger.info(f"Dataset: {dataset_path}")
logger.info(f"Products processed: {len(df)}")
logger.info(f"Embedding model: {embedding_model}")
logger.info(f"Embedding dimension: {vectors.shape[1]}")
logger.info(f"Output file: {embedding_file_path}.npy")
logger.info(f"File size: {file_size:.2f} MB")
logger.info("Embedding generation completed successfully!")
logger.info("=" * 50)