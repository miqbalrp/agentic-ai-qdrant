from qdrant_client import QdrantClient, models
import pandas as pd
import numpy as np
from uuid import uuid4
import time

# Initialize Qdrant client
qdrant_url = "http://localhost:6333"
client = QdrantClient(url=qdrant_url, timeout=60.0)
print("Qdrant client initialized.")

# Load dataset
dataset_path = "dataset/product_catalog.json"
df = pd.read_json(dataset_path)
print(f"Loaded dataset with {len(df)} products.")

# Load embeddings from file
embedding_file_path = "embeddings/product_catalog.npy"
vectors = np.load(embedding_file_path)
vector_dimension = vectors.shape[1]
print(f"Loaded embeddings with shape: {vectors.shape}")

qdrant_collection_name = "product_catalog"

# Check if collection exists and delete if it does
if client.collection_exists(qdrant_collection_name):
    print(f"Collection '{qdrant_collection_name}' already exists.")
    client.delete_collection(qdrant_collection_name)
    print(f"Deleted existing collection '{qdrant_collection_name}'.")

# Create a new collection
try:
    client.create_collection(
        collection_name=qdrant_collection_name,
        vectors_config=models.VectorParams(size=vector_dimension, distance=models.Distance.COSINE)
    )
    print(f"Created collection '{qdrant_collection_name}'.")
except Exception as ce:
    # Handle collection creation timeout
    if "timed out" in str(ce).lower():
        print("Timed out while creating collection. Polling for collection availability...")
        start_wait = time.time()
        while time.time() - start_wait < 60:
            try:
                if client.collection_exists(qdrant_collection_name):
                    print(f"Collection '{qdrant_collection_name}' is now available")
                    break
            except Exception as e:
                pass # Ignore errors while polling
            time.sleep(1)
        else:
            raise TimeoutError("Timed out waiting for collection to become available")
    else:
        raise

# Prepare points for insertion
points = []
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
            "description": row["description"],
            "url": row["url"]
        }
    )
    points.append(point)
print(f"Prepared {len(points)} points for insertion.")

# Insert points into the collection
client.upsert(
    collection_name=qdrant_collection_name,
    points=points,
    wait=True 
)

# Verify insertion by checking collection info
collection_info = client.get_collection(qdrant_collection_name)
points_count = collection_info.points_count
print(f"Inserted {points_count} points into collection '{qdrant_collection_name}'.")
