import open_clip
import torch
import numpy as np
import pandas as pd
from PIL import Image
import os
from pathlib import Path
from tqdm import tqdm
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
IMAGES_DIR = Path("../images")
EMBEDDINGS_DIR = Path("../embeddings")
BATCH_SIZE = 32
MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"

def setup_model():
    """Load the CLIP model and preprocessing pipeline."""
    logger.info(f"Loading CLIP model: {MODEL_NAME} with {PRETRAINED}")
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, 
        pretrained=PRETRAINED
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    logger.info(f"Model loaded on device: {device}")
    return model, preprocess, device

def get_image_files():
    """Get list of all image files and their corresponding object IDs."""
    image_files = []
    object_ids = []
    
    for img_path in IMAGES_DIR.glob("*.jpg"):
        object_id = int(img_path.stem)
        image_files.append(img_path)
        object_ids.append(object_id)
    
    # Sort by object ID for consistent ordering
    sorted_pairs = sorted(zip(object_ids, image_files))
    object_ids, image_files = zip(*sorted_pairs)
    
    logger.info(f"Found {len(image_files)} images")
    return list(object_ids), list(image_files)

def process_batch(model, preprocess, device, image_paths, object_ids):
    """Process a batch of images and return embeddings."""
    images = []
    valid_indices = []
    
    # Load and preprocess images
    for i, img_path in enumerate(image_paths):
        try:
            image = Image.open(img_path).convert('RGB')
            image_tensor = preprocess(image)
            images.append(image_tensor)
            valid_indices.append(i)
        except Exception as e:
            logger.warning(f"Failed to load {img_path}: {e}")
            continue
    
    if not images:
        return None, []
    
    # Stack images into batch tensor
    batch_tensor = torch.stack(images).to(device)
    
    # Generate embeddings
    with torch.no_grad():
        image_features = model.encode_image(batch_tensor)
        # Normalize embeddings
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
    # Get valid object IDs
    valid_object_ids = [object_ids[i] for i in valid_indices]
    
    return image_features.cpu().numpy(), valid_object_ids

def generate_embeddings():
    """Generate CLIP embeddings for all images."""
    # Create embeddings directory
    EMBEDDINGS_DIR.mkdir(exist_ok=True)
    
    # Setup model
    model, preprocess, device = setup_model()
    
    # Get all image files
    object_ids, image_files = get_image_files()
    
    all_embeddings = []
    all_object_ids = []
    
    # Process in batches
    for i in tqdm(range(0, len(image_files), BATCH_SIZE), desc="Processing batches"):
        batch_paths = image_files[i:i + BATCH_SIZE]
        batch_object_ids = object_ids[i:i + BATCH_SIZE]
        
        embeddings, valid_object_ids = process_batch(
            model, preprocess, device, batch_paths, batch_object_ids
        )
        
        if embeddings is not None:
            all_embeddings.append(embeddings)
            all_object_ids.extend(valid_object_ids)
    
    # Concatenate all embeddings
    if all_embeddings:
        final_embeddings = np.vstack(all_embeddings)
        final_object_ids = np.array(all_object_ids)
        
        # Save embeddings
        np.save(EMBEDDINGS_DIR / "clip_embeddings.npy", final_embeddings)
        np.save(EMBEDDINGS_DIR / "object_ids.npy", final_object_ids)
        
        # Save metadata
        metadata = {
            "model_name": MODEL_NAME,
            "pretrained": PRETRAINED,
            "embedding_dim": final_embeddings.shape[1],
            "num_embeddings": len(final_object_ids),
            "device_used": str(device)
        }
        
        with open(EMBEDDINGS_DIR / "clip_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Generated {len(final_object_ids)} embeddings")
        logger.info(f"Embedding dimension: {final_embeddings.shape[1]}")
        logger.info(f"Saved to {EMBEDDINGS_DIR}")
        
        return final_embeddings, final_object_ids
    else:
        logger.error("No embeddings were generated!")
        return None, None

if __name__ == "__main__":
    generate_embeddings()
