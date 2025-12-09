import pandas as pd
import requests
import json
import os
import time
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List
from tqdm import tqdm
import logging
from PIL import Image, ImageOps

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("download_log.txt"), logging.StreamHandler()],
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

# Directories
DATA_DIR = Path("../data")
IMAGES_DIR = Path("../images")
METADATA_DIR = Path("../metadata")
THUMBNAIL_DIR = Path("../thumbnails")
CSV_OUTPUT = METADATA_DIR / "artwork_metadata.csv"
BATCH_SIZE = 100

# Create directories
IMAGES_DIR.mkdir(exist_ok=True)
METADATA_DIR.mkdir(exist_ok=True)
THUMBNAIL_DIR.mkdir(exist_ok=True)


def get_artwork_from_api(
    object_id: int, max_retries: int = 3
) -> Optional[Dict[str, Any]]:
    """Fetch artwork details from Met Collection API with retry logic."""
    url = f"{API_BASE_URL}/{object_id}"

    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=HEADERS, timeout=10)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                logger.warning(f"Object {object_id} not found in API")
                return None
            elif response.status_code == 403:
                logger.error(f"403 Forbidden for object {object_id} - possible rate limit or access issue")
                wait_time = (attempt + 1) * 5
                logger.warning(f"Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            elif response.status_code == 429:
                # Rate limited, wait longer
                wait_time = (attempt + 1) * 2
                logger.warning(f"Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.warning(
                    f"API returned status {response.status_code} for object {object_id}"
                )
                return None
        except requests.RequestException as e:
            logger.warning(
                f"Request failed for object {object_id} (attempt {attempt + 1}/{max_retries}): {e}"
            )
            if attempt < max_retries - 1:
                time.sleep(1)

    return None


def download_image(image_url: str, object_id: int, max_retries: int = 3) -> bool:
    """Download image from URL and save to disk."""
    image_path = IMAGES_DIR / f"{object_id}.jpg"

    # Skip if already downloaded
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
                logger.warning(
                    f"Failed to download image for object {object_id}, status: {response.status_code}"
                )
                if response.status_code == 403:
                    wait_time = (attempt + 1) * 5
                    logger.warning(f"403 error on image download, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                return False
        except requests.RequestException as e:
            logger.warning(
                f"Download failed for object {object_id} (attempt {attempt + 1}/{max_retries}): {e}"
            )
            if attempt < max_retries - 1:
                time.sleep(1)

    return False


def save_metadata_json(
    object_id: int, excel_data: Dict[str, Any], api_data: Dict[str, Any]
):
    """Save combined metadata as JSON file."""
    metadata_path = METADATA_DIR / f"{object_id}.json"

    # Combine Excel and API data
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

def generate_thumbnails(max_side: int = 160, quality: int = 82):
    """
    Walk through the images folder and create JPEG thumbnails
    into THUMBNAIL_DIR, similar to make_thumbs.py.
    """
    THUMBNAIL_DIR.mkdir(exist_ok=True)

    logger.info("Starting thumbnail generation...")

    # Only look at .jpg files; ignore everything else
    image_files = sorted(IMAGES_DIR.glob("*.jpg"))

    for src in tqdm(image_files, desc="Creating thumbnails"):
        dst = THUMBNAIL_DIR / (src.stem + ".jpg")

        # If we already made this thumbnail, skip it
        if dst.exists():
            continue

        try:
            with Image.open(src) as im:
                # Fix weird EXIF rotations
                im = ImageOps.exif_transpose(im)

                # Make sure we are in a safe color mode
                if im.mode not in ("RGB", "L"):
                    im = im.convert("RGB")

                # Shrink the image so the longest side is max_side
                im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)

                # Save the thumbnail as a JPEG
                im.save(
                    dst,
                    format="JPEG",
                    quality=quality,
                    optimize=True,
                    progressive=True,
                )
        except Exception as e:
            logger.warning(f"Thumbnail failed for {src.name}: {e}")

    logger.info("Thumbnail generation finished.")


def prepare_csv_row(
    object_id: int, excel_row: pd.Series, api_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Prepare a row for the CSV output with all relevant metadata."""
    row = {
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

    return row


def get_existing_object_ids() -> set:
    """Get set of object IDs that already exist in CSV to prevent duplicates."""
    if not os.path.exists(CSV_OUTPUT):
        return set()
    
    try:
        existing_df = pd.read_csv(CSV_OUTPUT)
        return set(existing_df["Object ID"].astype(str))
    except Exception as e:
        logger.warning(f"Could not read existing CSV: {e}")
        return set()


def append_to_csv(new_rows: List[Dict[str, Any]]):
    """Append new rows to CSV file, creating header if file doesn't exist."""
    if not new_rows:
        return
    
    df_new = pd.DataFrame(new_rows)
    
    if os.path.exists(CSV_OUTPUT):
        # Append mode
        df_new.to_csv(CSV_OUTPUT, mode='a', header=False, index=False, encoding="utf-8")
    else:
        # Create new file with header
        df_new.to_csv(CSV_OUTPUT, index=False, encoding="utf-8")
    
    logger.info(f"Appended {len(new_rows)} rows to {CSV_OUTPUT}")


def choose_data_file() -> str:
    """Let user choose between available data files."""
    csv_file = DATA_DIR / "met_paintings.csv"
    excel_file = DATA_DIR / "MetObjects.xlsx"
    
    available_files = []
    if csv_file.exists():
        available_files.append(("CSV", str(csv_file)))
    if excel_file.exists():
        available_files.append(("Excel", str(excel_file)))
    
    if not available_files:
        raise FileNotFoundError("No data files found in data folder")
    
    if len(available_files) == 1:
        logger.info(f"Using {available_files[0][0]} file: {available_files[0][1]}")
        return available_files[0][1]
    
    print("\nMultiple data files found:")
    for i, (file_type, file_path) in enumerate(available_files, 1):
        print(f"{i}. {file_type}: {file_path}")
    
    while True:
        try:
            choice = input(f"\nChoose file (1-{len(available_files)}): ").strip()
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(available_files):
                selected_file = available_files[choice_idx][1]
                logger.info(f"Selected: {selected_file}")
                return selected_file
            else:
                print(f"Please enter a number between 1 and {len(available_files)}")
        except ValueError:
            print("Please enter a valid number")


def process_artworks(data_file: str, limit: Optional[int] = None):
    """Main processing function to download images and save metadata."""
    logger.info(f"Reading data file: {data_file}")
    
    # Read file based on extension
    if data_file.endswith('.csv'):
        df = pd.read_csv(data_file)
    else:
        try:
            df = pd.read_excel(data_file, engine='openpyxl')
        except:
            df = pd.read_excel(data_file, engine='xlrd')
    
    logger.info(f"Total artworks in file: {len(df)}")

    if limit:
        df = df.head(limit)
        logger.info(f"Processing limited to first {limit} artworks")

    # Get existing object IDs to prevent duplicates
    existing_ids = get_existing_object_ids()
    logger.info(f"Found {len(existing_ids)} existing records in CSV")

    # Track results
    downloaded_count = 0
    skipped_no_image = 0
    skipped_duplicate_image = 0
    skipped_existing = 0
    errors = []
    csv_rows = []
    downloaded_image_urls = set()  # Track unique image URLs

    # Process each artwork
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing artworks"):
        object_id = row.get("Object ID")

        if pd.isna(object_id):
            logger.warning(f"Row {idx} has no Object ID, skipping")
            continue

        object_id = int(object_id)
        object_id_str = str(object_id)

        # Skip if already in CSV
        if object_id_str in existing_ids:
            skipped_existing += 1
            logger.debug(f"Object {object_id} already exists in CSV, skipping")
            continue

        # Check if already downloaded
        image_path = IMAGES_DIR / f"{object_id}.jpg"
        metadata_path = METADATA_DIR / f"{object_id}.json"
        
        # Check if we already have both image and metadata
        if image_path.exists() and metadata_path.exists():
            logger.debug(f"Image and metadata {object_id} already exist, loading from cache")
            with open(metadata_path, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
                api_data = cached_data.get("api_metadata", {})
                # Track this URL to prevent duplicates
                cached_url = api_data.get("primaryImage", "")
                if cached_url:
                    downloaded_image_urls.add(cached_url)
                csv_rows.append(prepare_csv_row(object_id, row, api_data))
                downloaded_count += 1
        # If we have image but no metadata, get metadata from API
        elif image_path.exists() and not metadata_path.exists():
            logger.info(f"Image {object_id} exists but no metadata, fetching metadata from API")
            api_data = get_artwork_from_api(object_id)
            if api_data:
                save_metadata_json(object_id, row.to_dict(), api_data)
                csv_rows.append(prepare_csv_row(object_id, row, api_data))
                downloaded_count += 1
        else:
            # Get artwork details from API
            api_data = get_artwork_from_api(object_id)

            if not api_data:
                errors.append(f"Object {object_id}: Failed to fetch from API")
                continue

            # Check if image is available
            primary_image_url = api_data.get("primaryImage", "")

            if not primary_image_url:
                skipped_no_image += 1
                logger.debug(f"Object {object_id} has no image available")
                continue

            # Check for duplicate image URL
            if primary_image_url in downloaded_image_urls:
                skipped_duplicate_image += 1
                logger.debug(
                    f"Object {object_id} has duplicate image URL, skipping download"
                )
                # Still save metadata for this object
                save_metadata_json(object_id, row.to_dict(), api_data)
                csv_rows.append(prepare_csv_row(object_id, row, api_data))
                continue

            # Download image
            if download_image(primary_image_url, object_id):
                # Save metadata
                save_metadata_json(object_id, row.to_dict(), api_data)
                csv_rows.append(prepare_csv_row(object_id, row, api_data))
                downloaded_image_urls.add(primary_image_url)  # Track this URL
                downloaded_count += 1
                logger.info(
                    f"Successfully downloaded: {object_id} - {row.get('Title', 'N/A')}"
                )
            else:
                errors.append(f"Object {object_id}: Failed to download image")

        # Save CSV every BATCH_SIZE items
        if len(csv_rows) >= BATCH_SIZE:
            append_to_csv(csv_rows)
            csv_rows = []  # Clear the batch

        # Rate limiting - Met API allows 80 req/s, we use 10 req/s to be safe
        time.sleep(REQUEST_DELAY)

    # Save any remaining rows
    if csv_rows:
        append_to_csv(csv_rows)

    # Summary
    logger.info("=" * 60)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total artworks processed: {len(df)}")
    logger.info(f"Successfully downloaded: {downloaded_count}")
    logger.info(f"Skipped (no image): {skipped_no_image}")
    logger.info(f"Skipped (duplicate image): {skipped_duplicate_image}")
    logger.info(f"Skipped (already in CSV): {skipped_existing}")
    logger.info(f"Errors: {len(errors)}")
    logger.info("=" * 60)

    # Save errors to file
    if errors:
        with open("download_errors.txt", "w") as f:
            f.write("\n".join(errors))
        logger.info(f"Error details saved to download_errors.txt")

     # After finishing downloads, build thumbnails for all JPG images
    generate_thumbnails()


def main():
    parser = argparse.ArgumentParser(
        description="Download Met Museum artwork images and metadata (robust version)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of artworks to process (useful for testing)",
    )

    args = parser.parse_args()

    try:
        data_file = choose_data_file()
        process_artworks(data_file, args.limit)
    except FileNotFoundError as e:
        logger.error(str(e))
        return
    except KeyboardInterrupt:
        logger.info("Download interrupted by user")
        return


if __name__ == "__main__":
    main()
