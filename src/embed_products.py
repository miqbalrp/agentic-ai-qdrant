import pandas as pd
import numpy as np
import openai
import sys
import os

# Add parent directory to path to import config
import config as CONFIG

# Load the dataset
df = pd.read_json(CONFIG.DATASET_PATH)

# Initialize OpenAI client with API key from environment
openai_client = openai.Client(
    api_key=CONFIG.OPENAI_API_KEY
)

# Prepare text descriptions
texts = [f"{row.name}. {row.description} Material: {row.material}. Color: {row.color}." for row in df.itertuples()]

# Generate embeddings using OpenAI
embedding_model = CONFIG.EMBEDDING_MODEL
response = openai_client.embeddings.create(input=texts, model=embedding_model)
print(f"Generated {len(response.data)} embeddings with dimension {len(response.data[0].embedding)}")

# Extract only the embedding vectors (not the full response objects)
embeddings = [item.embedding for item in response.data]
vectors = np.array(embeddings)

# Save embeddings using config path
embeddings_path = CONFIG.EMBEDDING_FILE
print(f"Saving {vectors.shape[0]} embeddings with dimension {vectors.shape[1]}")
np.save(embeddings_path, vectors, allow_pickle=False)

print("Embeddings saved successfully!")