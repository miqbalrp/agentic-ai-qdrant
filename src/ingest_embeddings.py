from qdrant_client import QdrantClient, models
import pandas as pd
import numpy as np
from uuid import uuid4
import config as CONFIG

client = QdrantClient(url=CONFIG.QDRANT_URL)

df = pd.read_json(CONFIG.DATASET_PATH)
vectors = np.load(CONFIG.EMBEDDING_FILE)

collection_name = CONFIG.QDRANT_COLLECTION_NAME
vector_dimension = vectors.shape[1]  # Should be 1536 for OpenAI text-embedding-3-small

# Create collection (delete existing if present)
if client.collection_exists(collection_name):
    print(f"Collection '{collection_name}' already exists. Deleting...")
    client.delete_collection(collection_name)
    print(f"Collection '{collection_name}' deleted.")
print(f"Creating new collection '{collection_name}' with {vector_dimension} dimensions...")
client.create_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(size=vector_dimension, distance=models.Distance.COSINE),
)
print(f"Collection '{collection_name}' created successfully.")  

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
            "description": row["description"]
        }
    )
    points.append(point)

# Insert points into Qdrant
print("Inserting embeddings into Qdrant...")
client.upsert(
    collection_name=collection_name,
    points=points,
    wait=True  # Wait for the operation to complete
)
print(f"Successfully ingested {len(points)} embeddings into Qdrant collection '{collection_name}'")