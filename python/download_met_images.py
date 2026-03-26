import pandas as pd
import requests
import json
import time
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List
from tqdm import tqdm
import logging
import os

# Storing static paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
IMAGES_DIR = ROOT / "images"
METADATA_DIR = ROOT / "metadata"
CSV_OUTPUT = METADATA_DIR / "artwork_metadata.csv"
BATCH_SIZE = 100

IMAGES_DIR.mkdir(exist_ok=True)
METADATA_DIR.mkdir(exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(ROOT / "download_log.txt"),
        logging.StreamHandler()
    ],
)
logger = logging.getLogger(__name__)

# Met Collection API base URL
API_BASE_URL = "https://collectionapi.metmuseum.org/public/collection/v1/objects"

# Request headers - essential to avoid 403 errors
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# Rate limiting - Met API allows 80 requests/second, we'll use 10/second to be safe
REQUEST_DELAY = 0.1  # 10 requests per second

def get_artwork_from_api(object_id: int, max_retries: int = 3) -> Optional[Dict[str, Any]]:
    url = f"{API_BASE_URL}/{object_id}"

    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=HEADERS, timeout=10)
            if response.status_code == 200:
                return response.json()
            elif response.status_code in (404,):
                logger.warning(f"Object {object_id} not found")
                return None
            elif response.status_code in (403, 429):
                wait_time = (attempt + 1) * 5
                logger.warning(f"{response.status_code} for {object_id}, waiting {wait_time}s")
                time.sleep(wait_time)
            else:
                logger.warning(f"Status {response.status_code} for {object_id}")
                return None
        except requests.RequestException as e:
            logger.warning(f"Request failed for {object_id}: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)

    return None

def download_image(image_url: str, object_id: int, max_retries: int = 3) -> bool:
    image_path = IMAGES_DIR / f"{object_id}.jpg"
    if image_path.exists():
        return True

    for attempt in range(max_retries):
        try:
            response = requests.get(image_url, headers=HEADERS, timeout=30, stream=True)
            if response.status_code == 200:
                with open(image_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                return True
            else:
                logger.warning(f"Image {object_id} status {response.status_code}")
                if response.status_code == 403:
                    wait_time = (attempt + 1) * 5
                    time.sleep(wait_time)
                    continue
                return False
        except requests.RequestException as e:
            logger.warning(f"Image download failed {object_id}: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)

    return False

def save_metadata_json(object_id: int, excel_data: Dict[str, Any], api_data: Dict[str, Any]):
    metadata_path = METADATA_DIR / f"{object_id}.json"
    combined_metadata = {
        "object_id": object_id,
        "excel_metadata": excel_data,
        "api_metadata": {
            "primaryImage": api_data.get("primaryImage", ""),
            "additionalImages": api_data.get("additionalImages", []),
            "constituents": api_data.get("constituents", []),
            "measurements": api_data.get("measurements", []),
            "tags": api_data.get("tags", []),
        },
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(combined_metadata, f, indent=2, ensure_ascii=False)

def prepare_csv_row(object_id: int, excel_row: pd.Series, api_data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "Object ID": object_id,
        "Object Number": excel_row.get("Object Number", ""),
        "Title": excel_row.get("Title", ""),
        "Artist Display Name": excel_row.get("Artist Display Name", ""),
        "Artist Display Bio": excel_row.get("Artist Display Bio", ""),
        "Artist Nationality": excel_row.get("Artist Nationality", ""),
        "Object Date": excel_row.get("Object Date", ""),
        "Object Begin Date": excel_row.get("Object Begin Date", ""),
        "Object End Date": excel_row.get("Object End Date", ""),
        "Medium": excel_row.get("Medium", ""),
        "Dimensions": excel_row.get("Dimensions", ""),
        "Department": excel_row.get("Department", ""),
        "Culture": excel_row.get("Culture", ""),
        "Period": excel_row.get("Period", ""),
        "Dynasty": excel_row.get("Dynasty", ""),
        "Reign": excel_row.get("Reign", ""),
        "Credit Line": excel_row.get("Credit Line", ""),
        "AccessionYear": excel_row.get("AccessionYear", ""),
        "Is Public Domain": excel_row.get("Is Public Domain", ""),
        "Is Highlight": excel_row.get("Is Highlight", ""),
        "Classification": excel_row.get("Classification", ""),
        "Object Name": excel_row.get("Object Name", ""),
        "Link Resource": excel_row.get("Link Resource", ""),
        "Object Wikidata URL": excel_row.get("Object Wikidata URL", ""),
        "Artist Wikidata URL": excel_row.get("Artist Wikidata URL", ""),
        "Artist ULAN URL": excel_row.get("Artist ULAN URL", ""),
        "Tags": excel_row.get("Tags", ""),
        "Geography Type": excel_row.get("Geography Type", ""),
        "City": excel_row.get("City", ""),
        "State": excel_row.get("State", ""),
        "Country": excel_row.get("Country", ""),
        "Region": excel_row.get("Region", ""),
        "Repository": excel_row.get("Repository", ""),
        "Primary Image URL": api_data.get("primaryImage", ""),
        "Number of Additional Images": len(api_data.get("additionalImages", [])),
        "Image Downloaded": True,
    }

def get_existing_object_ids() -> set:
    if not CSV_OUTPUT.exists():
        return set()
    try:
        existing_df = pd.read_csv(CSV_OUTPUT)
        return set(existing_df["Object ID"].astype(str))
    except Exception as e:
        logger.warning(f"Could not read existing CSV: {e}")
        return set()

def append_to_csv(new_rows: List[Dict[str, Any]]):
    if not new_rows:
        return
    df_new = pd.DataFrame(new_rows)
    if CSV_OUTPUT.exists():
        df_new.to_csv(CSV_OUTPUT, mode='a', header=False, index=False, encoding="utf-8")
    else:
        df_new.to_csv(CSV_OUTPUT, index=False, encoding="utf-8")
    logger.info(f"Appended {len(new_rows)} rows to {CSV_OUTPUT}")

def load_input_table(path: Path) -> pd.DataFrame:
    logger.info(f"Reading source data: {path}")
    if str(path).endswith(".csv"):
        return pd.read_csv(path)
    else:
        # try modern engine first then fallback
        try:
            return pd.read_excel(path, engine='openpyxl')
        except:
            return pd.read_excel(path, engine='xlrd')

def process_artworks(data_file: Path, limit: Optional[int] = None):
    df = load_input_table(data_file)
    logger.info(f"Total rows: {len(df)}")
    if limit:
        df = df.head(limit)
        logger.info(f"Limiting to first {limit} rows")

    existing_ids = get_existing_object_ids()
    logger.info(f"Existing IDs in CSV: {len(existing_ids)}")

    downloaded_count = 0
    skipped_no_image = 0
    skipped_duplicate_image = 0
    skipped_existing = 0
    csv_rows = []
    downloaded_image_urls = set()
    errors = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing artworks"):
        object_id = row.get("Object ID")
        if pd.isna(object_id):
            continue
        object_id = int(object_id)
        object_id_str = str(object_id)

        # skip if already recorded in CSV
        if object_id_str in existing_ids:
            skipped_existing += 1
            continue

        image_path = IMAGES_DIR / f"{object_id}.jpg"
        metadata_path = METADATA_DIR / f"{object_id}.json"

        if image_path.exists() and metadata_path.exists():
            # load cached metadata JSON
            with open(metadata_path, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
            api_data = cached_data.get("api_metadata", {})
            cached_url = api_data.get("primaryImage", "")
            if cached_url:
                downloaded_image_urls.add(cached_url)
            csv_rows.append(prepare_csv_row(object_id, row, api_data))
            downloaded_count += 1

        else:
            # fetch fresh
            api_data = get_artwork_from_api(object_id)
            if not api_data:
                errors.append(f"Object {object_id}: API fail")
                continue

            primary_image_url = api_data.get("primaryImage", "")
            if not primary_image_url:
                skipped_no_image += 1
                continue

            if primary_image_url in downloaded_image_urls:
                skipped_duplicate_image += 1
                # still save metadata
                save_metadata_json(object_id, row.to_dict(), api_data)
                csv_rows.append(prepare_csv_row(object_id, row, api_data))
            else:
                if download_image(primary_image_url, object_id):
                    save_metadata_json(object_id, row.to_dict(), api_data)
                    csv_rows.append(prepare_csv_row(object_id, row, api_data))
                    downloaded_image_urls.add(primary_image_url)
                    downloaded_count += 1
                else:
                    errors.append(f"Object {object_id}: image download failed")

        # flush batch
        if len(csv_rows) >= BATCH_SIZE:
            append_to_csv(csv_rows)
            csv_rows = []

        time.sleep(REQUEST_DELAY)

    if csv_rows:
        append_to_csv(csv_rows)

    logger.info("SUMMARY")
    logger.info(f"Downloaded: {downloaded_count}")
    logger.info(f"No image: {skipped_no_image}")
    logger.info(f"Duplicate image: {skipped_duplicate_image}")
    logger.info(f"Already existed: {skipped_existing}")
    logger.info(f"Errors: {len(errors)}")

    if errors:
        with open(ROOT / "download_errors.txt", "w") as f:
            f.write("\n".join(errors))
        logger.info("Wrote download_errors.txt")

def main():
    parser = argparse.ArgumentParser(description="Download Met images + build metadata/json/CSV")
    parser.add_argument("--limit", type=int, default=None, help="Limit rows (testing)")
    parser.add_argument("--file", type=str, default=None,
                        help="Path to input data file. If omitted, we'll auto-pick met_paintings.csv or MetObjects.xlsx")
    args = parser.parse_args()

    if args.file:
        data_file = Path(args.file)
    else:
        # auto-pick
        if (DATA_DIR / "met_paintings.csv").exists():
            data_file = DATA_DIR / "met_paintings.csv"
        elif (DATA_DIR / "MetObjects.xlsx").exists():
            data_file = DATA_DIR / "MetObjects.xlsx"
        else:
            raise FileNotFoundError("No input data found in data/")

    process_artworks(data_file, args.limit)

if __name__ == "__main__":
    main()