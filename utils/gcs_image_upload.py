from google.cloud import storage
import os

# Get the parent directory (project root)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Set up authentication with correct path
CREDENTIALS_PATH = os.path.join(PROJECT_ROOT, 'keys', 'gcs-credentials.json')
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = CREDENTIALS_PATH

# Initialize the GCS client
def get_storage_client():
    return storage.Client()

# Upload a single image
def upload_image_to_gcs(local_file_path, bucket_name, destination_blob_name):
    """
    Uploads an image file to Google Cloud Storage.
    
    Args:
        local_file_path: Path to the local image file (e.g., 'images/photo.jpg')
        bucket_name: Name of your GCS bucket (e.g., 'my-app-images-12345')
        destination_blob_name: Path in GCS where file will be stored (e.g., 'uploads/photo.jpg')
    """
    try:
        client = get_storage_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        
        # Upload the file
        blob.upload_from_filename(local_file_path)
        
        print(f"✓ File {local_file_path} uploaded to gs://{bucket_name}/{destination_blob_name}")
        
        # Return the public URL (if bucket is public) or the gs:// URL
        return f"gs://{bucket_name}/{destination_blob_name}"
        
    except Exception as e:
        print(f"✗ Error uploading {local_file_path}: {e}")
        return None

# Upload all images from a folder
def upload_folder_to_gcs(local_folder, bucket_name, gcs_folder_prefix=""):
    """
    Uploads all images from a local folder to GCS.
    
    Args:
        local_folder: Local folder path (e.g., 'images/')
        bucket_name: Name of your GCS bucket
        gcs_folder_prefix: Folder prefix in GCS (e.g., 'uploads/')
    """
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
    
    if not os.path.exists(local_folder):
        print(f"✗ Error: Folder '{local_folder}' not found")
        return
    
    for filename in os.listdir(local_folder):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            local_path = os.path.join(local_folder, filename)
            gcs_path = os.path.join(gcs_folder_prefix, filename).replace('\\', '/')
            
            upload_image_to_gcs(local_path, bucket_name, gcs_path)

# Example usage
if __name__ == "__main__":
    BUCKET_NAME = "met-images-storage"
    
    # Build correct path to images folder
    IMAGES_FOLDER = os.path.join(PROJECT_ROOT, 'images')
    
    # Upload all images from the images folder
    upload_folder_to_gcs(
        local_folder=IMAGES_FOLDER,
        bucket_name=BUCKET_NAME,
        gcs_folder_prefix="uploads"
    )