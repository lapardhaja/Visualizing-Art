import pandas as pd
import requests
import time
import logging
import os

# Configure logging
logging.basicConfig(filename="errors.log", level=logging.WARNING,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Load your CSV
df = pd.read_csv("summary_combined_reduced_with_links.csv")

# Create new columns if they don't exist
for col in ["query_image", "similar_image"]:
    if col not in df.columns:
        df[col] = ""

# Cache to avoid repeated requests
cache = {}

def get_image_url(object_id):
    """Fetch the primaryImageSmall from the Met API, with caching and error handling."""
    if pd.isna(object_id):
        return ""
    
    object_id = str(object_id)
    if object_id in cache:
        return cache[object_id]

    api_url = f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{object_id}"
    try:
        r = requests.get(api_url, timeout=5)
        if r.ok:
            data = r.json()
            url = data.get("primaryImageSmall", "")
            cache[object_id] = url
            return url
        else:
            logging.warning(f"Request failed for {object_id}: {r.status_code}")
    except Exception as e:
        logging.warning(f"Error fetching {object_id}: {e}")
    
    cache[object_id] = ""
    return ""

# Function to process a column safely and resumable
def fetch_images_for_column(df, id_col, img_col, start_idx=0):
    for i in range(start_idx, len(df)):
        if df.at[i, img_col] == "":
            obj_id = df.at[i, id_col]
            df.at[i, img_col] = get_image_url(obj_id)
            time.sleep(0.1)  # avoid API throttling
        # Save progress every 100 rows
        if i % 100 == 0 and i != start_idx:
            df.to_csv("with_images_progress.csv", index=False)
    df.to_csv("with_images_final.csv", index=False)

# Fetch images for both columns
fetch_images_for_column(df, "query_object_id", "query_image")
fetch_images_for_column(df, "similar_object_id", "similar_image")

print("Done! Saved to with_images_final.csv")
