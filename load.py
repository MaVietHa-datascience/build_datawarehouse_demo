from minio import Minio
import os

def initialize_minio_client(endpoint="localhost:9000", access_key="minioadmin", secret_key="minioadmin", secure=False):
    """Initializes and returns a MinIO client."""
    client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)
    return client

def create_bucket_if_not_exists(client, bucket_name):
    """Creates a MinIO bucket if it doesn't already exist."""
    try:
        if not client.bucket_exists(bucket_name):
            client.make_bucket(bucket_name)
            print(f"Bucket '{bucket_name}' created successfully.")
        else:
            print(f"Bucket '{bucket_name}' already exists.")
    except Exception as e:
        print(f"Error creating/checking bucket '{bucket_name}': {e}")
        return False
    return True

def upload_files_to_minio(client, local_folder, bucket_name, object_prefix, file_extension):
    """Uploads files with a specific extension from a local folder to a MinIO bucket."""
    files_to_upload = [f for f in os.listdir(local_folder) if f.endswith(file_extension)]
    for file_name in files_to_upload:
        local_file_path = os.path.join(local_folder, file_name)
        object_name = f"{object_prefix}{file_name}"
        try:
            client.fput_object(bucket_name, object_name, local_file_path)
            print(f"Uploaded '{local_file_path}' to '{bucket_name}/{object_name}'")
        except Exception as e:
            print(f"Error uploading '{local_file_path}': {e}")

def main():
    """Main function to initialize MinIO and upload files."""
    # MinIO connection details
    minio_endpoint = "localhost:9000"
    minio_access_key = "minioadmin"
    minio_secret_key = "minioadmin"
    minio_secure = False

    # Bucket name
    bucket_name = "yelpdataset"

    # Local folder paths
    json_folder = r"D:\Data_AE_Test\Data\Yelp JSON"  # Raw string for Windows path
    csv_folder = r"D:\Data_AE_Test\Data\Climate Data"  # Raw string for Windows path

    # Initialize MinIO client
    minio_client = initialize_minio_client(minio_endpoint, minio_access_key, minio_secret_key, minio_secure)
    if not minio_client:
        return

    # Create bucket if it doesn't exist
    if not create_bucket_if_not_exists(minio_client, bucket_name):
        return

    # Upload JSON files
    upload_files_to_minio(minio_client, json_folder, bucket_name, "raw/yelpjson/", ".json")

    # Upload CSV files
    upload_files_to_minio(minio_client, csv_folder, bucket_name, "raw/climatedata/", ".csv")

if __name__ == "__main__":
    main()