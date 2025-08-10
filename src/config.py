import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY in environment variables")

# Qdrant Configuration
QDRANT_URL = "http://localhost:6333"
QDRANT_COLLECTION_NAME = "product_catalog"

# File Paths
DATASET_PATH = "dataset/product_catalog.json"
EMBEDDING_FILE = "embeddings/product_catalog.npy"

# Embedding Model Configuration
EMBEDDING_MODEL = "text-embedding-3-small"

# Dataset Categories and Brands
PRODUCT_CATEGORIES = [
    "dresses",
    "pants", 
    "shirts",
    "sweaters",
    "t-shirts",
    "skirts",
    "jackets"
]

PRODUCT_BRANDS = [
    "Zara",
    "Levi's", 
    "H&M",
    "Uniqlo",
    "Adidas"
]