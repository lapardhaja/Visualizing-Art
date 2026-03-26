import numpy as np
import json
import pandas as pd
from pathlib import Path
import logging
from sklearn.preprocessing import normalize

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
EMBEDDINGS_DIR = ROOT / "embeddings"
METADATA_CSV = ROOT / "metadata" / "artwork_metadata.csv"

def load_clip_embeddings():
    logger.info("Loading CLIP embeddings")
    clip_embeddings = np.load(EMBEDDINGS_DIR / "clip_embeddings.npy")
    object_ids = np.load(EMBEDDINGS_DIR / "object_ids.npy")
    logger.info(f"CLIP embeddings: {clip_embeddings.shape}")
    logger.info(f"Object IDs: {len(object_ids)}")
    return clip_embeddings, object_ids

def load_metadata_features():
    logger.info("Loading metadata features")
    metadata_features = np.load(EMBEDDINGS_DIR / "metadata_features.npy")
    with open(EMBEDDINGS_DIR / "feature_names.json", "r") as f:
        feature_names = json.load(f)
    logger.info(f"Metadata features: {metadata_features.shape}")
    logger.info(f"Num metadata feature names: {len(feature_names)}")
    return metadata_features, feature_names

def load_metadata_dataframe():
    df = pd.read_csv(METADATA_CSV)
    return df

def create_object_id_mapping(clip_object_ids, metadata_df):
    logger.info("Creating ID mapping between CLIP and metadata features")
    metadata_object_ids = metadata_df['Object ID'].values

    id_to_indices = {}

    for i, obj_id in enumerate(clip_object_ids):
        d = id_to_indices.setdefault(int(obj_id), {})
        d['clip_index'] = i

    for i, obj_id in enumerate(metadata_object_ids):
        d = id_to_indices.setdefault(int(obj_id), {})
        d['metadata_index'] = i

    common_ids = [
        obj_id for obj_id, idxs in id_to_indices.items()
        if 'clip_index' in idxs and 'metadata_index' in idxs
    ]

    logger.info(f"Common objects w/ both CLIP and metadata: {len(common_ids)}")
    return common_ids, id_to_indices

def combine_embeddings(clip_embeddings, metadata_features, common_ids, id_to_indices):
    logger.info("Concatenating CLIP + metadata features per object")
    combined_embeddings = []
    combined_object_ids = []

    for obj_id in common_ids:
        clip_idx = id_to_indices[obj_id]['clip_index']
        meta_idx = id_to_indices[obj_id]['metadata_index']

        clip_vec = clip_embeddings[clip_idx]
        meta_vec = metadata_features[meta_idx]

        combo = np.concatenate([clip_vec, meta_vec])
        combined_embeddings.append(combo)
        combined_object_ids.append(obj_id)

    combined_embeddings = np.array(combined_embeddings)
    combined_object_ids = np.array(combined_object_ids)

    logger.info(f"Combined embedding matrix shape: {combined_embeddings.shape}")
    return combined_embeddings, combined_object_ids

def normalize_embeddings(embeddings):
    logger.info("L2-normalizing combined embeddings for cosine similarity")
    return normalize(embeddings, norm='l2')

def save_combined_features(combined_embeddings, combined_object_ids, metadata_feature_names):
    EMBEDDINGS_DIR.mkdir(exist_ok=True)

    np.save(EMBEDDINGS_DIR / "combined_embeddings.npy", combined_embeddings)
    np.save(EMBEDDINGS_DIR / "combined_object_ids.npy", combined_object_ids)

    # CLIP is 512 dims (ViT-B/32), metadata_feature_names is rest
    clip_feature_names = [f"clip_{i}" for i in range(512)]
    metadata_feature_names_prefixed = [f"meta_{name}" for name in metadata_feature_names]
    all_feature_names = clip_feature_names + metadata_feature_names_prefixed

    with open(EMBEDDINGS_DIR / "combined_feature_names.json", "w") as f:
        json.dump(all_feature_names, f, indent=2)

    combination_metadata = {
        "total_embeddings": len(combined_object_ids),
        "embedding_dimension": combined_embeddings.shape[1],
        "clip_dimension": 512,
        "metadata_dimension": len(metadata_feature_names),
        "feature_breakdown": {
            "clip_features": len(clip_feature_names),
            "metadata_features": len(metadata_feature_names)
        }
    }
    with open(EMBEDDINGS_DIR / "combination_metadata.json", "w") as f:
        json.dump(combination_metadata, f, indent=2)

    logger.info("Saved combined features and metadata")

def combine_features():
    clip_embeddings, clip_object_ids = load_clip_embeddings()
    metadata_features, metadata_feature_names = load_metadata_features()
    metadata_df = load_metadata_dataframe()

    common_ids, id_map = create_object_id_mapping(clip_object_ids, metadata_df)
    combined, combined_ids = combine_embeddings(clip_embeddings, metadata_features, common_ids, id_map)
    normalized = normalize_embeddings(combined)
    save_combined_features(normalized, combined_ids, metadata_feature_names)

    return normalized, combined_ids, metadata_feature_names

if __name__ == "__main__":
    combine_features()
