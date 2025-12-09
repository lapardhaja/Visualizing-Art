import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def add_metadata():
    """Add metadata to summary_cross_artist.csv by merging with MetObjects.xlsx."""
    # Get paths relative to project root (one level up from python/)
    project_root = Path(__file__).parent.parent
    summary_path = project_root / 'embeddings' / 'summary_cross_artist.csv'
    metobjects_path = project_root / 'data' / 'MetObjects.xlsx'
    output_path = project_root / 'embeddings' / 'summary_cross_artist_with_metadata.csv'
    
    logger.info(f"Loading summary file: {summary_path}")
    df1 = pd.read_csv(summary_path)
    
    logger.info(f"Loading MetObjects file: {metobjects_path}")
    df2 = pd.read_excel(metobjects_path, dtype={'Object ID': str})
    
    # Also make sure df1 IDs are strings for consistent merging
    df1['query_object_id'] = df1['query_object_id'].astype(str)
    df1['similar_object_id'] = df1['similar_object_id'].astype(str)
    
    # Columns to keep
    cols_to_keep = [
        'Object ID', 'Object Name', 'Title', 'Object Date', 'Object Begin Date', 'Object End Date',
        'Artist Display Name', 'Artist Nationality', 'Artist Begin Date', 'Artist End Date',
        'Artist Gender', 'Artist Wikidata URL', 'Culture', 'Country', 'City', 'Region',
        'Department', 'Classification', 'Medium', 'Link Resource', 'Object Wikidata URL', 'Tags'
    ]
    
    # ---------- Merge for query_object_id ----------
    logger.info("Merging metadata for query objects...")
    df_query = df2[cols_to_keep].add_prefix('query_')
    df_query = df_query.rename(columns={'query_Object ID': 'query_object_id'})
    df1 = df1.merge(df_query, on='query_object_id', how='left')
    
    # ---------- Merge for similar_object_id ----------
    logger.info("Merging metadata for similar objects...")
    df_similar = df2[cols_to_keep].add_prefix('similar_')
    df_similar = df_similar.rename(columns={'similar_Object ID': 'similar_object_id'})
    df1 = df1.merge(df_similar, on='similar_object_id', how='left')
    
    # ---------- Convert all columns to snake_case (final step only) ----------
    logger.info("Converting column names to snake_case...")
    df1.columns = (
        df1.columns
        .str.strip()
        .str.lower()
        .str.replace(r'\s+', '_', regex=True)       # spaces → underscores
        .str.replace(r'[^0-9a-zA-Z_]', '', regex=True)  # remove special characters
    )
    
    # ---------- Save final file ----------
    logger.info(f"Saving output to: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df1.to_csv(output_path, index=False)
    logger.info("Metadata added successfully")

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    add_metadata()
