import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import json
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Static file paths
ROOT = Path(__file__).resolve().parents[1]
METADATA_CSV = ROOT / "metadata" / "artwork_metadata.csv"
EMBEDDINGS_DIR = ROOT / "embeddings"

def load_metadata():
    logger.info(f"Loading metadata from {METADATA_CSV}")
    df = pd.read_csv(METADATA_CSV)
    logger.info(f"Loaded {len(df)} records")
    return df
def clean_metadata(df):
    df_clean = df.copy()

    # Fill missing values for categorical-ish columns
    categorical_cols = [
        'Artist Display Name', 'Medium', 'Department', 'Country', 'Period', 'Culture'
    ]
    for col in categorical_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna('Unknown')

    # Tags for text modeling
    df_clean['Tags'] = df_clean.get('Tags', '').fillna('')
    df_clean['Tags_clean'] = df_clean['Tags'].str.replace('|', ' ', regex=False)

    # Title for text modeling
    df_clean['Title'] = df_clean['Title'].fillna('')

    # numeric-ish date columns
    date_cols = ['Object Begin Date', 'Object End Date', 'AccessionYear']
    for col in date_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            df_clean[f'{col}_missing'] = df_clean[col].isna().astype(int)
            df_clean[col] = df_clean[col].fillna(0)

    return df_clean

def extract_categorical_features(df_clean):
    logger.info("Extracting categorical features")

    categorical_cols = [
        'Artist Display Name',
        'Medium',
        'Department',
        'Country',
        'Period',
        'Culture'
    ]

    all_encoded_blocks = []
    all_feature_names = []

    for col in categorical_cols:
        if col not in df_clean.columns:
            logger.warning(f"Missing column {col}, skipping")
            continue

        enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded = enc.fit_transform(df_clean[[col]])

        cats = enc.categories_[0]
        col_feature_names = [f"{col}_{c}" for c in cats]

        all_encoded_blocks.append(encoded)
        all_feature_names.extend(col_feature_names)

        logger.info(f"{col}: {encoded.shape[1]} one-hot features")

    if all_encoded_blocks:
        cat_features = np.hstack(all_encoded_blocks)
    else:
        cat_features = np.zeros((len(df_clean), 0))

    return cat_features, all_feature_names

def extract_numerical_features(df_clean):
    logger.info("Extracting numerical features")
    num_cols = ['Object Begin Date', 'Object End Date', 'AccessionYear']
    miss_cols = [f"{c}_missing" for c in num_cols]

    all_cols = []
    for c in num_cols:
        if c in df_clean.columns:
            all_cols.append(c)
    for c in miss_cols:
        if c in df_clean.columns:
            all_cols.append(c)

    if not all_cols:
        return np.zeros((len(df_clean), 0)), []

    # normalize only the non-missing-indicator numeric columns
    norm_cols = [c for c in num_cols if c in df_clean.columns]
    scaled_blocks = []
    feature_names = []

    # scale numeric cols
    if norm_cols:
        scaler = StandardScaler()
        scaled_numeric = scaler.fit_transform(df_clean[norm_cols])
        scaled_blocks.append(scaled_numeric)
        feature_names.extend([f"num_{c}" for c in norm_cols])

    # add missing indicator cols (0/1), unscaled
    for mcol in miss_cols:
        if mcol in df_clean.columns:
            scaled_blocks.append(df_clean[[mcol]].values)
            feature_names.append(f"flag_{mcol}")

    numerical_features = np.hstack(scaled_blocks) if scaled_blocks else np.zeros((len(df_clean), 0))
    logger.info(f"Numerical feature dim: {numerical_features.shape[1]}")
    return numerical_features, feature_names

def extract_text_features(df_clean):
    logger.info("Extracting text features")
    df_clean['combined_text'] = df_clean['Title'] + ' ' + df_clean['Tags_clean']

    tfidf = TfidfVectorizer(
        max_features=100,
        stop_words='english',
        ngram_range=(1,2),
        min_df=2,
        max_df=0.8
    )

    tfidf_matrix = tfidf.fit_transform(df_clean['combined_text']).toarray()
    feat_names = [f"tfidf_{t}" for t in tfidf.get_feature_names_out()]
    logger.info(f"TF-IDF dim: {tfidf_matrix.shape[1]}")
    return tfidf_matrix, feat_names

def assemble_all_features(cat_features, cat_names, num_features, num_names, text_features, text_names):
    logger.info("Combining all feature blocks")

    blocks = []
    names = []

    if cat_features.shape[1] > 0:
        blocks.append(cat_features)
        names.extend(cat_names)

    if num_features.shape[1] > 0:
        blocks.append(num_features)
        names.extend(num_names)

    if text_features.shape[1] > 0:
        blocks.append(text_features)
        names.extend(text_names)

    if blocks:
        combined = np.hstack(blocks)
    else:
        combined = np.zeros((len(cat_features), 0))

    logger.info(f"Final metadata feature matrix: {combined.shape}")
    logger.info(f"Total feature columns: {len(names)}")
    return combined, names

def save_features(features, feature_names, object_ids):
    EMBEDDINGS_DIR.mkdir(exist_ok=True)

    np.save(EMBEDDINGS_DIR / "metadata_features.npy", features)

    with open(EMBEDDINGS_DIR / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)

    info = {
        "num_features": features.shape[1],
        "num_samples": features.shape[0],
        "feature_types_hint": {
            "categorical+onehot+text+numeric": "mixed"
        }
    }
    with open(EMBEDDINGS_DIR / "metadata_processing_info.json", "w") as f:
        json.dump(info, f, indent=2)

    logger.info(f"Saved metadata features to {EMBEDDINGS_DIR}")

def process_metadata():
    df = load_metadata()
    df_clean = clean_metadata(df)

    cat_feats, cat_names = extract_categorical_features(df_clean)
    num_feats, num_names = extract_numerical_features(df_clean)
    text_feats, text_names = extract_text_features(df_clean)

    combined_feats, all_names = assemble_all_features(
        cat_feats, cat_names,
        num_feats, num_names,
        text_feats, text_names
    )

    save_features(combined_feats, all_names, df_clean['Object ID'].values)

    return combined_feats, all_names, df_clean

if __name__ == "__main__":
    process_metadata()