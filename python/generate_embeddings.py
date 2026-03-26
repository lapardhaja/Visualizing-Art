import open_clip
import torch
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import json
import logging

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
IMAGES_DIR = ROOT / "images"
EMBEDDINGS_DIR = ROOT / "embeddings"

BATCH_SIZE = 32
MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"

def setup_model():
    logger.info(f"Loading CLIP {MODEL_NAME} / {PRETRAINED}")
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=PRETRAINED
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    logger.info(f"Using device: {device}")
    return model, preprocess, device

def get_image_files():
    image_files = []
    object_ids = []
    for img_path in IMAGES_DIR.glob("*.jpg"):
        try:
            object_id = int(img_path.stem)
        except ValueError:
            continue
        image_files.append(img_path)
        object_ids.append(object_id)

    pairs = sorted(zip(object_ids, image_files))
    object_ids, image_files = zip(*pairs) if pairs else ([], [])
    logger.info(f"Found {len(image_files)} images")
    return list(object_ids), list(image_files)

def process_batch(model, preprocess, device, image_paths, object_ids):
    images = []
    valid_indices = []

    for i, img_path in enumerate(image_paths):
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = preprocess(img)
            images.append(img_tensor)
            valid_indices.append(i)
        except Exception as e:
            logger.warning(f"Failed to load {img_path}: {e}")

    if not images:
        return None, []

    batch_tensor = torch.stack(images).to(device)

    with torch.no_grad():
        feats = model.encode_image(batch_tensor)
        feats = feats / feats.norm(dim=-1, keepdim=True)

    valid_object_ids = [object_ids[i] for i in valid_indices]
    return feats.cpu().numpy(), valid_object_ids

def generate_embeddings():
    EMBEDDINGS_DIR.mkdir(exist_ok=True)

    model, preprocess, device = setup_model()
    object_ids, image_files = get_image_files()

    all_embeddings = []
    all_ids = []

    for i in tqdm(range(0, len(image_files), BATCH_SIZE), desc="Embedding batches"):
        batch_paths = image_files[i:i+BATCH_SIZE]
        batch_ids = object_ids[i:i+BATCH_SIZE]

        emb, valid_ids = process_batch(model, preprocess, device, batch_paths, batch_ids)
        if emb is not None:
            all_embeddings.append(emb)
            all_ids.extend(valid_ids)

    if not all_embeddings:
        logger.error("No embeddings generated.")
        return None, None

    final_embeddings = np.vstack(all_embeddings)
    final_object_ids = np.array(all_ids)

    np.save(EMBEDDINGS_DIR / "clip_embeddings.npy", final_embeddings)
    np.save(EMBEDDINGS_DIR / "object_ids.npy", final_object_ids)

    meta = {
        "model_name": MODEL_NAME,
        "pretrained": PRETRAINED,
        "embedding_dim": int(final_embeddings.shape[1]),
        "num_embeddings": int(len(final_object_ids)),
        "device_used": str(device)
    }
    with open(EMBEDDINGS_DIR / "clip_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Saved {len(final_object_ids)} embeddings -> {EMBEDDINGS_DIR}")
    return final_embeddings, final_object_ids

if __name__ == "__main__":
    generate_embeddings()
