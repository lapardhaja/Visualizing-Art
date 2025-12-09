import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import json
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import warnings

from config import (
    METADATA_CSV, EMBEDDINGS_DIR, METADATA_CONFIG, PIPELINE_CONFIG
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_metadata():
    """Load and clean the metadata CSV."""
    logger.info(f"Loading metadata from {METADATA_CSV}")
    df = pd.read_csv(METADATA_CSV)
    logger.info(f"Loaded {len(df)} records")
    return df

def clean_metadata(df: pd.DataFrame, 
                  unknown_strategy: str = None) -> pd.DataFrame:
    """Clean and prepare metadata for feature extraction.
    
    Args:
        df: Raw metadata DataFrame
        unknown_strategy: Strategy for handling unknown values
                         (defaults to METADATA_CONFIG.UNKNOWN_HANDLING_STRATEGY)
    
    Returns:
        Cleaned DataFrame with appropriate unknown handling
    """
    # Create a copy to avoid modifying original
    df_clean = df.copy()
    
    if unknown_strategy is None:
        unknown_strategy = METADATA_CONFIG.UNKNOWN_HANDLING_STRATEGY
    
    # Track original missing values for each categorical column
    categorical_cols = METADATA_CONFIG.CATEGORICAL_COLUMNS
    
    # Store which values were originally missing
    for col in categorical_cols:
        if col in df_clean.columns:
            df_clean[f'__{col}_was_missing'] = df_clean[col].isna()
    
    # Handle missing values based on strategy
    if unknown_strategy == METADATA_CONFIG.UNKNOWN_HANDLING_CATEGORY:
        # Legacy behavior: fill with 'Unknown'
        for col in categorical_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna('Unknown')
                
    elif unknown_strategy in [METADATA_CONFIG.UNKNOWN_HANDLING_ZERO, 
                             METADATA_CONFIG.UNKNOWN_HANDLING_MISSING_INDICATOR]:
        # New behavior: keep NaN for now, will handle in encoding
        pass
    else:
        raise ValueError(f"Unknown handling strategy: {unknown_strategy}")
    
    # Clean Tags column - split by | and join with spaces for TF-IDF
    df_clean['Tags'] = df_clean['Tags'].fillna('')
    df_clean['Tags_clean'] = df_clean['Tags'].str.replace('|', ' ', regex=False)
    
    # Clean Title for TF-IDF
    df_clean['Title'] = df_clean['Title'].fillna('')
    
    # Handle date columns
    date_cols = METADATA_CONFIG.NUMERICAL_COLUMNS
    for col in date_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            # Create missing indicator before filling
            df_clean[f'{col}_missing'] = df_clean[col].isna().astype(int)
            # Fill missing dates with 0
            df_clean[col] = df_clean[col].fillna(0)
    
    # Calculate metadata completeness for each artwork
    n_categorical = len([c for c in categorical_cols if c in df_clean.columns])
    if n_categorical > 0:
        completeness = 0
        for col in categorical_cols:
            if col in df_clean.columns:
                completeness += (~df_clean[f'__{col}_was_missing']).astype(int)
        df_clean['metadata_completeness'] = completeness / n_categorical
    else:
        df_clean['metadata_completeness'] = 1.0
    
    return df_clean

def extract_categorical_features(df: pd.DataFrame,
                               unknown_strategy: str = None) -> Tuple[Dict, List]:
    """Extract and encode categorical features with configurable unknown handling.
    
    Args:
        df: Cleaned DataFrame with categorical columns
        unknown_strategy: How to handle unknown/missing values
        
    Returns:
        Tuple of (categorical_features dict, feature_names list)
    """
    logger.info("Extracting categorical features")
    
    if unknown_strategy is None:
        unknown_strategy = METADATA_CONFIG.UNKNOWN_HANDLING_STRATEGY
    
    categorical_features = {}
    feature_names = []
    missing_indicators = []
    
    categorical_cols = METADATA_CONFIG.CATEGORICAL_COLUMNS
    
    for col in categorical_cols:
        if col not in df.columns:
            logger.warning(f"Column {col} not found, skipping")
            continue
        
        # Create a working copy of the column
        working_col = df[col].copy()
        was_missing = df[f'__{col}_was_missing']
        
        if unknown_strategy == METADATA_CONFIG.UNKNOWN_HANDLING_ZERO:
            # Strategy 1: Remove NaN values before encoding
            # This will create zero vectors for missing values
            valid_mask = ~working_col.isna()
            valid_values = working_col[valid_mask]
            
            if len(valid_values) > 0:
                # Fit encoder only on non-missing values
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoder.fit(valid_values.values.reshape(-1, 1))
                
                # Transform all values, NaN will become zero vectors
                encoded = np.zeros((len(df), len(encoder.categories_[0])))
                if valid_mask.any():
                    encoded[valid_mask] = encoder.transform(
                        working_col[valid_mask].values.reshape(-1, 1)
                    )
            else:
                # All values are missing
                logger.warning(f"Column {col} has no valid values")
                encoded = np.zeros((len(df), 1))
                encoder = None
                
        elif unknown_strategy == METADATA_CONFIG.UNKNOWN_HANDLING_MISSING_INDICATOR:
            # Strategy 2: Zero vectors + explicit missing indicators
            valid_mask = ~working_col.isna()
            valid_values = working_col[valid_mask]
            
            if len(valid_values) > 0:
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoder.fit(valid_values.values.reshape(-1, 1))
                
                encoded = np.zeros((len(df), len(encoder.categories_[0])))
                if valid_mask.any():
                    encoded[valid_mask] = encoder.transform(
                        working_col[valid_mask].values.reshape(-1, 1)
                    )
            else:
                encoded = np.zeros((len(df), 1))
                encoder = None
            
            # Add missing indicator
            missing_ind = was_missing.astype(int).values.reshape(-1, 1)
            missing_indicators.append(missing_ind)
            feature_names.append(f"{col}_is_missing")
            
        else:  # UNKNOWN_HANDLING_CATEGORY (legacy)
            # Fill NaN with 'Unknown' and encode normally
            working_col = working_col.fillna('Unknown')
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoded = encoder.fit_transform(working_col.values.reshape(-1, 1))
        
        # Create feature names (excluding 'Unknown' if using zero vectors)
        if encoder is not None:
            categories = encoder.categories_[0]
            if unknown_strategy != METADATA_CONFIG.UNKNOWN_HANDLING_CATEGORY:
                # Don't create feature for 'Unknown' category if it exists
                categories = [cat for cat in categories if cat != 'Unknown']
            
            feature_names_col = [f"{col}_{cat}" for cat in categories]
            feature_names.extend(feature_names_col)
        else:
            feature_names_col = []
        
        categorical_features[col] = {
            'encoded': encoded,
            'feature_names': feature_names_col,
            'encoder': encoder,
            'n_missing': was_missing.sum()
        }
        
        logger.info(f"Encoded {col}: {encoded.shape[1]} features, "
                   f"{was_missing.sum()} missing values")
    
    # Add missing indicators if using that strategy
    if missing_indicators:
        categorical_features['_missing_indicators'] = {
            'encoded': np.hstack(missing_indicators),
            'feature_names': [f for f in feature_names 
                            if f.endswith('_is_missing')],
            'encoder': None
        }
    
    return categorical_features, feature_names

def extract_numerical_features(df):
    """Extract and normalize numerical features."""
    logger.info("Extracting numerical features")
    
    numerical_cols = ['Object Begin Date', 'Object End Date', 'AccessionYear']
    missing_cols = [f'{col}_missing' for col in numerical_cols]
    all_numerical_cols = numerical_cols + missing_cols
    
    numerical_features = []
    feature_names = []
    
    for col in all_numerical_cols:
        if col in df.columns:
            # For missing indicators, don't normalize (they're already 0/1)
            if col.endswith('_missing'):
                numerical_features.append(df[[col]].values)
                feature_names.append(col)
                logger.info(f"Added missing indicator: {col}")
            else:
                # Normalize the actual date features
                scaler = StandardScaler()
                normalized = scaler.fit_transform(df[[col]])
                numerical_features.append(normalized)
                feature_names.append(col)
                logger.info(f"Normalized {col}")
    
    if numerical_features:
        numerical_features = np.hstack(numerical_features)
    else:
        numerical_features = np.array([]).reshape(len(df), 0)
    
    return numerical_features, feature_names

def extract_text_features(df: pd.DataFrame) -> Tuple[np.ndarray, List[str], TfidfVectorizer]:
    """Extract TF-IDF features from text fields."""
    logger.info("Extracting text features")
    
    # Combine Title and Tags for richer text representation
    df['combined_text'] = df['Title'] + ' ' + df['Tags_clean']
    
    # TF-IDF vectorization with configuration
    tfidf = TfidfVectorizer(
        max_features=METADATA_CONFIG.TFIDF_MAX_FEATURES,
        stop_words='english',
        ngram_range=METADATA_CONFIG.TFIDF_NGRAM_RANGE,
        min_df=METADATA_CONFIG.TFIDF_MIN_DF,
        max_df=METADATA_CONFIG.TFIDF_MAX_DF
    )
    
    tfidf_features = tfidf.fit_transform(df['combined_text']).toarray()
    feature_names = [f"tfidf_{term}" for term in tfidf.get_feature_names_out()]
    
    logger.info(f"Extracted {tfidf_features.shape[1]} TF-IDF features")
    
    return tfidf_features, feature_names, tfidf

def combine_features(categorical_features, numerical_features, text_features):
    """Combine all feature types into a single matrix."""
    logger.info("Combining all features")
    
    all_features = []
    all_feature_names = []
    
    # Add categorical features
    for col, data in categorical_features.items():
        all_features.append(data['encoded'])
        all_feature_names.extend(data['feature_names'])
    
    # Add numerical features
    if numerical_features.size > 0:
        all_features.append(numerical_features)
        all_feature_names.extend([f"num_{name}" for name in ['Object Begin Date', 'Object End Date', 'AccessionYear'] if name in ['Object Begin Date', 'Object End Date', 'AccessionYear']])
    
    # Add text features
    if text_features.size > 0:
        all_features.append(text_features)
        all_feature_names.extend([f"tfidf_{name}" for name in range(text_features.shape[1])])
    
    # Combine all features
    if all_features:
        combined_features = np.hstack(all_features)
    else:
        # Get number of samples from any component (they should all have same length)
        n_samples = 0
        if numerical_features.size > 0:
            n_samples = numerical_features.shape[0]
        elif text_features.size > 0:
            n_samples = text_features.shape[0]
        elif categorical_features:
            # Get from first categorical feature
            first_cat = next(iter(categorical_features.values()))
            n_samples = first_cat['encoded'].shape[0]
        
        combined_features = np.array([]).reshape(n_samples, 0)
    
    logger.info(f"Combined features shape: {combined_features.shape}")
    logger.info(f"Total feature names: {len(all_feature_names)}")
    
    return combined_features, all_feature_names

def save_features(features: np.ndarray, 
                 feature_names: List[str], 
                 object_ids: np.ndarray,
                 metadata_completeness: Optional[np.ndarray] = None) -> None:
    """Save processed features and metadata."""
    logger.info("Saving processed features")
    
    # Create embeddings directory
    EMBEDDINGS_DIR.mkdir(exist_ok=True)
    
    # Save features
    np.save(EMBEDDINGS_DIR / "metadata_features.npy", features)
    
    # Save feature names
    with open(EMBEDDINGS_DIR / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)
    
    # Save metadata completeness if provided
    if metadata_completeness is not None:
        np.save(EMBEDDINGS_DIR / "metadata_completeness.npy", metadata_completeness)
        logger.info(f"Average metadata completeness: {metadata_completeness.mean():.2%}")
    
    # Save metadata about the processing
    metadata = {
        "num_features": features.shape[1],
        "num_samples": features.shape[0],
        "feature_types": {
            "categorical": len([f for f in feature_names if not f.startswith(('num_', 'tfidf_'))]),
            "numerical": len([f for f in feature_names if f.startswith('num_')]),
            "text": len([f for f in feature_names if f.startswith('tfidf_')])
        }
    }
    
    with open(EMBEDDINGS_DIR / "metadata_processing_info.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved features to {EMBEDDINGS_DIR}")
    logger.info(f"Feature breakdown: {metadata['feature_types']}")

def validate_zero_vectors(features: np.ndarray, 
                         feature_names: List[str],
                         df: pd.DataFrame) -> Dict[str, any]:
    """Validate that unknowns produce zero vectors as expected.
    
    Returns:
        Validation results dictionary
    """
    results = {'passed': True, 'issues': []}
    
    if METADATA_CONFIG.UNKNOWN_HANDLING_STRATEGY != METADATA_CONFIG.UNKNOWN_HANDLING_ZERO:
        logger.info("Skipping zero vector validation (not using zero vector strategy)")
        return results
    
    logger.info("Validating zero vectors for unknown values")
    
    # Check a few examples where we know values are missing
    sample_size = min(10, len(df))
    sample_indices = np.random.choice(len(df), sample_size, replace=False)
    
    for idx in sample_indices:
        row = df.iloc[idx]
        for col in METADATA_CONFIG.CATEGORICAL_COLUMNS:
            if col in df.columns and row[f'__{col}_was_missing']:
                # Find features for this column
                col_features = [i for i, name in enumerate(feature_names) 
                              if name.startswith(f"{col}_") and not name.endswith("_is_missing")]
                
                if col_features:
                    # Check if all are zero
                    feature_values = features[idx, col_features]
                    if np.any(feature_values != 0):
                        issue = f"Row {idx} has missing {col} but non-zero features"
                        results['issues'].append(issue)
                        results['passed'] = False
    
    if results['passed']:
        logger.info("[OK] Zero vector validation passed")
    else:
        logger.warning(f"[X] Zero vector validation failed: {len(results['issues'])} issues")
        for issue in results['issues'][:5]:  # Show first 5 issues
            logger.warning(f"  - {issue}")
    
    return results


def process_metadata(unknown_strategy: str = None,
                    validate: bool = None) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    """Main function to process metadata and extract features.
    
    Args:
        unknown_strategy: How to handle unknown values (defaults to config)
        validate: Whether to validate the encoding (defaults to config)
        
    Returns:
        Tuple of (features, feature_names, cleaned_dataframe)
    """
    if unknown_strategy is None:
        unknown_strategy = METADATA_CONFIG.UNKNOWN_HANDLING_STRATEGY
    
    if validate is None:
        validate = PIPELINE_CONFIG.VALIDATE_ZERO_VECTORS
    
    logger.info(f"Processing metadata with unknown strategy: {unknown_strategy}")
    
    # Load and clean data
    df = load_metadata()
    df_clean = clean_metadata(df, unknown_strategy=unknown_strategy)
    
    # Extract different types of features
    categorical_features, cat_feature_names = extract_categorical_features(
        df_clean, unknown_strategy=unknown_strategy
    )
    numerical_features, num_feature_names = extract_numerical_features(df_clean)
    text_features, text_feature_names, tfidf = extract_text_features(df_clean)
    
    # Combine all features
    combined_features, all_feature_names = combine_features(
        categorical_features, numerical_features, text_features
    )
    
    # Validate if requested
    if validate:
        validation_results = validate_zero_vectors(
            combined_features, all_feature_names, df_clean
        )
        
    # Save everything
    save_features(combined_features, all_feature_names, 
                 df_clean['Object ID'].values, df_clean['metadata_completeness'].values)
    
    # Save metadata about the processing strategy
    processing_info = {
        "num_features": combined_features.shape[1],
        "num_samples": combined_features.shape[0],
        "unknown_handling_strategy": unknown_strategy,
        "feature_types": {
            "categorical": len([f for f in all_feature_names if not f.startswith(('num_', 'tfidf_'))]),
            "numerical": len([f for f in all_feature_names if f.startswith('num_')]),
            "text": len([f for f in all_feature_names if f.startswith('tfidf_')])
        },
        "validation_results": validation_results if validate else None
    }
    
    with open(EMBEDDINGS_DIR / "metadata_processing_info.json", "w") as f:
        json.dump(processing_info, f, indent=2)
    
    return combined_features, all_feature_names, df_clean

if __name__ == "__main__":
    process_metadata()
