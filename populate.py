# IMPORTS
import torch
from torchvision import models, transforms
from PIL import Image
import os
import pandas as pd
from pydantic import BaseModel
from transformers import pipeline, BitsAndBytesConfig
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


# Configuration for quantization to reduce VRAM usage
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

# Load the llava model for image description
model_id = "llava-hf/llava-1.5-7b-hf"
pipe = pipeline("image-to-text", model=model_id, model_kwargs={"quantization_config": quantization_config})

# Initialize text embedding model
embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
    device="cuda"
)

# Function to generate image description
def describe_image(image_path, pipe, prompt, max_new_tokens=200):
    outputs = pipe(image_path, prompt=prompt, generate_kwargs={"max_new_tokens": max_new_tokens})
    # Extract the relevant part of the generated text
    generated_text = outputs[0]["generated_text"]
    description = generated_text.split('ASSISTANT:')[1].strip()
    return description

# Function to get text embeddings
def get_text_embedding(text, embed_model):
    return embed_model.get_text_embedding(text)

# Directory containing images
image_folder = './images'
image_ids = []
descriptions = []
embeddings = []

# Process each image in the folder
for image_name in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_name)
    if os.path.isfile(image_path) and image_name.endswith(('.png', '.jpg', '.jpeg')):
        # Describe the image
        prompt = "USER: <image>\nDescribe the picture in general.\nASSISTANT:"
        description = describe_image(image_path, pipe, prompt)
        # Get text embedding
        embedding = get_text_embedding(description, embed_model)
        # Store results
        image_ids.append(image_name)
        descriptions.append(description)
        embeddings.append(embedding)
        print (f"Image: {image_name}, Description: {description} \n end of image \n")

# Convert lists to DataFrame
embeddings_df = pd.DataFrame(embeddings)
embeddings_df.insert(0, 'description', descriptions)
embeddings_df.insert(0, 'image_id', image_ids)

# Save to CSV
embeddings_df.to_csv('image_text_embeddings.csv', index=False)

# Load from CSV (for verification)
loaded_df = pd.read_csv('image_text_embeddings.csv')
print(loaded_df)