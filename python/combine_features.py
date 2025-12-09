import numpy as np
import json
from pathlib import Path
import logging
from sklearn.preprocessing import normalize
from typing import Dict, Tuple, Optional
import pandas as pd

from config import (
    EMBEDDINGS_DIR, METADATA_CONFIG, CLIP_EMBEDDING_DIM, SIMILARITY_CONFIG
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_clip_embeddings():
    """Load CLIP embeddings and object IDs."""
    logger.info("Loading CLIP embeddings")
    
    clip_embeddings = np.load(EMBEDDINGS_DIR / "clip_embeddings.npy")
    object_ids = np.load(EMBEDDINGS_DIR / "object_ids.npy")
    
    logger.info(f"Loaded CLIP embeddings: {clip_embeddings.shape}")
    logger.info(f"Loaded object IDs: {len(object_ids)}")
    
    return clip_embeddings, object_ids

def load_metadata_features():
    """Load processed metadata features."""
    logger.info("Loading metadata features")
    
    metadata_features = np.load(EMBEDDINGS_DIR / "metadata_features.npy")
    
    with open(EMBEDDINGS_DIR / "feature_names.json", "r") as f:
        feature_names = json.load(f)
    
    logger.info(f"Loaded metadata features: {metadata_features.shape}")
    logger.info(f"Feature names: {len(feature_names)}")
    
    return metadata_features, feature_names

def create_object_id_mapping(clip_object_ids, metadata_df):
    """Create mapping between object IDs and feature indices."""
    logger.info("Creating object ID to index mapping")
    
    # Get object IDs from metadata
    metadata_object_ids = metadata_df['Object ID'].values
    
    # Create mapping: object_id -> (clip_index, metadata_index)
    id_to_indices = {}
    
    # Map CLIP embeddings
    for i, obj_id in enumerate(clip_object_ids):
        if obj_id not in id_to_indices:
            id_to_indices[obj_id] = {}
        id_to_indices[obj_id]['clip_index'] = i
    
    # Map metadata features
    for i, obj_id in enumerate(metadata_object_ids):
        if obj_id not in id_to_indices:
            id_to_indices[obj_id] = {}
        id_to_indices[obj_id]['metadata_index'] = i
    
    # Find common object IDs (those with both CLIP and metadata)
    common_ids = []
    for obj_id, indices in id_to_indices.items():
        if 'clip_index' in indices and 'metadata_index' in indices:
            common_ids.append(obj_id)
    
    logger.info(f"Found {len(common_ids)} objects with both CLIP and metadata")
    
    return common_ids, id_to_indices

def combine_embeddings(clip_embeddings: np.ndarray, 
                      metadata_features: np.ndarray, 
                      common_ids: list, 
                      id_to_indices: dict,
                      visual_weight: float = None,
                      metadata_weight: float = None,
                      metadata_completeness: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Combine CLIP embeddings with metadata features using configurable weighting.
    
    Args:
        clip_embeddings: CLIP visual embeddings
        metadata_features: Metadata feature vectors
        common_ids: Object IDs present in both datasets
        id_to_indices: Mapping from object ID to indices
        visual_weight: Weight for visual features (defaults to config)
        metadata_weight: Weight for metadata features (defaults to config)
        metadata_completeness: Completeness scores for adaptive weighting
        
    Returns:
        Tuple of (combined_embeddings, combined_object_ids)
    """
    logger.info("Combining CLIP embeddings with metadata features")
    
    if visual_weight is None:
        visual_weight = METADATA_CONFIG.VISUAL_WEIGHT
    if metadata_weight is None:
        metadata_weight = METADATA_CONFIG.METADATA_WEIGHT
        
    logger.info(f"Using weights - Visual: {visual_weight}, Metadata: {metadata_weight}")
    
    combined_embeddings = []
    combined_object_ids = []
    completeness_scores = []
    
    for obj_id in common_ids:
        clip_idx = id_to_indices[obj_id]['clip_index']
        metadata_idx = id_to_indices[obj_id]['metadata_index']
        
        # Get embeddings
        clip_embedding = clip_embeddings[clip_idx]
        metadata_feature = metadata_features[metadata_idx]
        
        # Apply weights
        weighted_clip = clip_embedding * visual_weight
        weighted_metadata = metadata_feature * metadata_weight
        
        # If we have completeness scores, use them for adaptive weighting
        if metadata_completeness is not None:
            completeness = metadata_completeness[metadata_idx]
            completeness_scores.append(completeness)
            
            # Adaptive weighting: reduce metadata weight for incomplete data
            # Full completeness (1.0) = full metadata weight
            # Zero completeness (0.0) = minimal metadata weight (10% of configured)
            adaptive_metadata_weight = metadata_weight * (0.1 + 0.9 * completeness)
            weighted_metadata = metadata_feature * adaptive_metadata_weight
            
        # Concatenate: [weighted CLIP | weighted metadata]
        combined_embedding = np.concatenate([weighted_clip, weighted_metadata])
        combined_embeddings.append(combined_embedding)
        combined_object_ids.append(obj_id)
    
    combined_embeddings = np.array(combined_embeddings)
    combined_object_ids = np.array(combined_object_ids)
    
    logger.info(f"Combined embeddings shape: {combined_embeddings.shape}")
    if completeness_scores:
        avg_completeness = np.mean(completeness_scores)
        logger.info(f"Average metadata completeness: {avg_completeness:.2%}")
    
    return combined_embeddings, combined_object_ids

def normalize_embeddings(embeddings):
    """L2-normalize embeddings for cosine similarity."""
    logger.info("Normalizing embeddings for cosine similarity")
    
    normalized_embeddings = normalize(embeddings, norm='l2')
    
    logger.info("Embeddings normalized")
    
    return normalized_embeddings

def save_combined_features(combined_embeddings: np.ndarray, 
                         combined_object_ids: np.ndarray, 
                         clip_feature_names: list,
                         metadata_feature_names: list,
                         visual_weight: float = 1.0,
                         metadata_weight: float = 1.0):
    """Save combined features and metadata."""
    logger.info("Saving combined features")
    
    # Save combined embeddings
    np.save(EMBEDDINGS_DIR / "combined_embeddings.npy", combined_embeddings)
    np.save(EMBEDDINGS_DIR / "combined_object_ids.npy", combined_object_ids)
    
    # Create comprehensive feature names
    all_feature_names = []
    
    # Add CLIP feature names
    clip_feature_names = [f"clip_{i}" for i in range(CLIP_EMBEDDING_DIM)]
    all_feature_names.extend(clip_feature_names)
    
    # Add metadata feature names with prefix
    metadata_feature_names_prefixed = [f"meta_{name}" for name in metadata_feature_names]
    all_feature_names.extend(metadata_feature_names_prefixed)
    
    # Save feature names
    with open(EMBEDDINGS_DIR / "combined_feature_names.json", "w") as f:
        json.dump(all_feature_names, f, indent=2)
    
    # Save metadata about the combination
    combination_metadata = {
        "total_embeddings": len(combined_object_ids),
        "embedding_dimension": combined_embeddings.shape[1],
        "clip_dimension": CLIP_EMBEDDING_DIM,
        "metadata_dimension": len(metadata_feature_names),
        "feature_breakdown": {
            "clip_features": len(clip_feature_names),
            "metadata_features": len(metadata_feature_names)
        },
        "weights": {
            "visual_weight": visual_weight,
            "metadata_weight": metadata_weight
        },
        "normalization": "L2",
        "combination_method": "weighted_concatenation"
    }
    
    with open(EMBEDDINGS_DIR / "combination_metadata.json", "w") as f:
        json.dump(combination_metadata, f, indent=2)
    
    logger.info(f"Saved combined features to {EMBEDDINGS_DIR}")
    logger.info(f"Combination metadata: {combination_metadata}")

def load_metadata_dataframe():
    """Load the original metadata CSV to get object IDs."""
    import pandas as pd
    metadata_csv = Path("../metadata/artwork_metadata.csv")
    df = pd.read_csv(metadata_csv)
    return df

def combine_features(visual_weight: float = None,
                    metadata_weight: float = None):
    """Main function to combine CLIP embeddings with metadata features.
    
    Args:
        visual_weight: Weight for visual features
        metadata_weight: Weight for metadata features
    """
    # Load CLIP embeddings
    clip_embeddings, clip_object_ids = load_clip_embeddings()
    
    # Load metadata features
    metadata_features, metadata_feature_names = load_metadata_features()
    
    # Try to load metadata completeness scores
    metadata_completeness = None
    completeness_path = EMBEDDINGS_DIR / "metadata_completeness.npy"
    if completeness_path.exists():
        logger.info("Loading metadata completeness scores")
        metadata_completeness = np.load(completeness_path)
    
    # Load original metadata for object ID mapping
    metadata_df = load_metadata_dataframe()
    
    # Create object ID mapping
    common_ids, id_to_indices = create_object_id_mapping(clip_object_ids, metadata_df)
    
    # Combine embeddings with weighting
    combined_embeddings, combined_object_ids = combine_embeddings(
        clip_embeddings, metadata_features, common_ids, id_to_indices,
        visual_weight=visual_weight,
        metadata_weight=metadata_weight,
        metadata_completeness=metadata_completeness
    )
    
    # Normalize for cosine similarity
    normalized_embeddings = normalize_embeddings(combined_embeddings)
    
    # Save everything
    save_combined_features(
        normalized_embeddings, 
        combined_object_ids, 
        [],  # CLIP feature names will be generated
        metadata_feature_names,
        visual_weight=visual_weight or METADATA_CONFIG.VISUAL_WEIGHT,
        metadata_weight=metadata_weight or METADATA_CONFIG.METADATA_WEIGHT
    )
    
    return normalized_embeddings, combined_object_ids, metadata_feature_names

if __name__ == "__main__":
    combine_features()
