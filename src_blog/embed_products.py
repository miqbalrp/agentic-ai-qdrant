import pandas as pd
import numpy as np
import openai

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_client = openai.Client(api_key=openai_api_key)
print("OpenAI client initialized.")

# Load dataset
dataset_path = "dataset/product_catalog.json"
df = pd.read_json(dataset_path)
print(f"Loaded dataset with {len(df)} products.")

# Prepare texts for embedding
texts = [f"{row.name}. {row.description} Material: {row.material}. Color: {row.color}." for row in df.itertuples()]
print(f"Number of texts to embed: {len(texts)}")

# Generate embeddings
embedding_model = "text-embedding-3-small"
response = openai_client.embeddings.create(input=texts, model=embedding_model)
print(f"Generated {len(response.data)} embeddings.")

# Extract embeddings and convert to numpy array
embeddings = [item.embedding for item in response.data]
vectors = np.array(embeddings)
print(f"Shape of embedding vectors: {vectors.shape}")

# Save embeddings to file
embedding_file_path = "embeddings/product_catalog.npy"
np.save(embedding_file_path, vectors, allow_pickle=False)
print(f"Embeddings saved to {embedding_file_path}.")