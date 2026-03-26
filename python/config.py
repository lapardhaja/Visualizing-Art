"""
Configuration for the artwork embedding and similarity system.
"""
from pathlib import Path

# Paths
DATA_DIR = Path("..")
IMAGES_DIR = DATA_DIR / "images"
METADATA_DIR = DATA_DIR / "metadata"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
METADATA_CSV = METADATA_DIR / "artwork_metadata.csv"

# CLIP Model Configuration
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"
CLIP_BATCH_SIZE = 32
CLIP_EMBEDDING_DIM = 512

# Metadata Processing Configuration
class MetadataConfig:
    """Configuration for metadata feature extraction."""
    
    # Unknown value handling strategies
    UNKNOWN_HANDLING_ZERO = "zero_vector"  # Unknown values get zero vector (recommended)
    UNKNOWN_HANDLING_CATEGORY = "category"  # Unknown as a category (legacy behavior)
    UNKNOWN_HANDLING_MISSING_INDICATOR = "missing_indicator"  # Zero vector + missing indicator
    
    # Default strategy
    UNKNOWN_HANDLING_STRATEGY = UNKNOWN_HANDLING_ZERO
    
    # Categorical columns to encode
    CATEGORICAL_COLUMNS = [
        'Artist Display Name', 
        'Medium', 
        'Department', 
        'Country', 
        'Period', 
        'Culture'
    ]
    
    # High-missing columns (>90% missing) that might benefit from special handling
    HIGH_MISSING_COLUMNS = ['Country', 'Period']
    
    # Numerical columns
    NUMERICAL_COLUMNS = ['Object Begin Date', 'Object End Date', 'AccessionYear']
    
    # Text processing
    TFIDF_MAX_FEATURES = 100
    TFIDF_NGRAM_RANGE = (1, 2)
    TFIDF_MIN_DF = 2
    TFIDF_MAX_DF = 0.8
    
    # Feature weighting (for balanced combination)
    VISUAL_WEIGHT = 1.0  # Weight for CLIP features
    METADATA_WEIGHT = 0.5  # Weight for metadata features (can be tuned)
    
    # Filtering options
    MAX_UNKNOWN_FIELDS = None  # Set to integer to filter artworks with too many unknowns
    MIN_METADATA_COMPLETENESS = 0.0  # Minimum fraction of non-unknown metadata fields

# Similarity Search Configuration
class SimilarityConfig:
    """Configuration for similarity search."""
    
    # Default search parameters
    DEFAULT_TOP_K = 5
    DEFAULT_MIN_SIMILARITY = 0.0
    
    # Confidence scoring based on metadata completeness
    ENABLE_CONFIDENCE_SCORES = True
    CONFIDENCE_WEIGHT_KNOWN_FIELDS = 0.1  # Bonus per known metadata field
    
    # Results filtering
    FILTER_LOW_CONFIDENCE = False
    MIN_CONFIDENCE_THRESHOLD = 0.3

# Pipeline Configuration
class PipelineConfig:
    """Configuration for the processing pipeline."""
    
    # Processing options
    FORCE_REGENERATE = False  # Force regeneration even if files exist
    VERBOSE = True  # Verbose logging
    
    # Validation
    VALIDATE_EMBEDDINGS = True  # Validate embeddings after generation
    VALIDATE_ZERO_VECTORS = True  # Ensure unknowns produce zero vectors
    
    # Backwards compatibility
    SAVE_LEGACY_FORMAT = False  # Also save in old format for compatibility

# Export configurations
METADATA_CONFIG = MetadataConfig()
SIMILARITY_CONFIG = SimilarityConfig()
PIPELINE_CONFIG = PipelineConfig()
