import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import gradio as gr
from PIL import Image
import os
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Initialize text embedding model
embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
    device="cuda"
)

# Load embeddings and metadata from CSV
loaded_df = pd.read_csv('image_text_embeddings.csv')

# Extract image IDs, descriptions, and embeddings
image_ids = loaded_df['image_id'].values
descriptions = loaded_df['description'].values
embeddings = loaded_df.iloc[:, 2:].values

# Process query image
def get_text_embedding(text, embed_model):
    return embed_model.get_text_embedding(text)

# Find similar images based on text similarity
def find_similar_images(query, embeddings, image_ids, embed_model, top_k=5):
    query_embedding = get_text_embedding(query, embed_model)
    query_embedding = np.array(query_embedding).reshape(1, -1).tolist()
    similarities = cosine_similarity(query_embedding, embeddings)
    similar_indices = np.argsort(similarities[0])[::-1][:top_k]
    return image_ids[similar_indices]

def display_images(image_ids, image_folder='./images'):
    images = []
    for image_id in image_ids:
        image_path = os.path.join(image_folder, image_id)
        images.append(Image.open(image_path))
    return images

def search_and_display(query, embed_model, embeddings, image_ids, image_folder='./images', top_k=1):
    similar_image_ids = find_similar_images(query, embeddings, image_ids, embed_model, top_k=top_k)
    return display_images(similar_image_ids, image_folder)



# Define the Gradio interface
def search_images(query):
    images = search_and_display(query, embed_model, embeddings, image_ids)
    return images

iface = gr.Interface(fn=search_images, inputs="text", outputs="gallery")

# Launch the Gradio app
iface.launch()

