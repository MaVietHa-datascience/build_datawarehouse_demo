import pandas as pd
import duckdb
from minio import Minio
from io import BytesIO
import json
import concurrent.futures
import logging
from functools import partial
import numpy as np
from datetime import datetime, timedelta

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('yelp_etl')

# --- Configuration ---
MINIO_ENDPOINT = 'localhost:9000'  # Replace with your MinIO endpoint if different
MINIO_ACCESS_KEY = 'minioadmin'        # Replace with your MinIO access key
MINIO_SECRET_KEY = 'minioadmin'        # Replace with your MinIO secret key
MINIO_BUCKET_NAME = 'yelpdataset'
JSON_RAW_PREFIX = 'raw/yelpjson/'
CSV_RAW_PREFIX = 'raw/climatedata/'
DUCKDB_DB_FILE = 'yelp_dw.db'

# File names
BUSINESS_FILE = 'yelp_academic_dataset_business.json'
CHECKIN_FILE = 'yelp_academic_dataset_checkin.json'
COVID_FILE = 'yelp_academic_dataset_covid_features.json'
REVIEW_FILE = 'yelp_academic_dataset_review.json'
USER_FILE = 'yelp_academic_dataset_user.json'
TIP_FILE = 'yelp_academic_dataset_tip.json'
TEMPERATURE_FILE = 'temperature-degreef.csv'
PRECIPITATION_FILE = 'las-vegas-mccarran-intl-ap-precipitation-inch.csv'

# Processing configuration
CHUNK_SIZE = 100000  # Number of rows to process at once
MAX_WORKERS = 4      # Maximum number of parallel workers

# --- MinIO Client ---
def get_minio_client():
    """Create and return a MinIO client with connection pooling."""
    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False  # Set to True if your MinIO uses HTTPS
    )

# --- Helper Functions ---
def read_json_from_minio(file_name, chunk_size=CHUNK_SIZE):
    """Read JSON data from MinIO in chunks to reduce memory usage."""
    file_path = f"{JSON_RAW_PREFIX}{file_name}"
    minio_client = get_minio_client()
    
    try:
        response = minio_client.get_object(MINIO_BUCKET_NAME, file_path)
        # For smaller files, read all at once
        # Removed CHECKIN_FILE from this list to enable chunking for it
        if file_name in [COVID_FILE, TIP_FILE]:
            return pd.read_json(BytesIO(response.read()), lines=True)

        # For larger files, use chunking
        chunks = []
        for chunk in pd.read_json(BytesIO(response.read()), lines=True, chunksize=chunk_size):
            chunks.append(chunk)
        
        if chunks:
            return pd.concat(chunks, ignore_index=True)
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error reading {file_path} from MinIO: {e}")
        return None

def read_csv_from_minio(file_name):
    """Read CSV data from MinIO."""
    file_path = f"{CSV_RAW_PREFIX}{file_name}"
    minio_client = get_minio_client()
    
    try:
        response = minio_client.get_object(MINIO_BUCKET_NAME, file_path)
        return pd.read_csv(BytesIO(response.read()), encoding="utf-8")
    except Exception as e:
        logger.error(f"Error reading {file_path} from MinIO: {e}")
        return None

def transform_covid_features(df):
    """Transform COVID features data."""
    if df.empty:
        return pd.DataFrame(columns=[
            'business_id', 'grubhub_enabled', 'request_a_quote_enabled',
            'covid_banner', 'temporary_closed_until', 'virtual_services_offered'
        ])
    
    # Use a more efficient approach without apply
    fact_covid_features = df[['business_id', 'Grubhub enabled', 
                             'Request a Quote Enabled', 'Covid Banner',
                             'Temporary Closed Until', 'Virtual Services Offered']].rename(
        columns={
            'Grubhub enabled': 'grubhub_enabled',
            'Request a Quote Enabled': 'request_a_quote_enabled',
            'Covid Banner': 'covid_banner',
            'Temporary Closed Until': 'temporary_closed_until',
            'Virtual Services Offered': 'virtual_services_offered'
        }
    ).drop_duplicates()

    # Extract highlights data
    highlights_data = []
    for _, row in df[['business_id', 'highlights']].iterrows():
        business_id = row['business_id']
        highlights = row['highlights']
        if isinstance(highlights, str):
            try:
                highlights = json.loads(highlights)
            except json.JSONDecodeError:
                highlights = []  # Handle invalid JSON
        if isinstance(highlights, list):
            for highlight in highlights:
                highlights_data.append({
                    'business_id': business_id,
                    'identifier': highlight.get('identifier'),
                    'params': highlight.get('params'),
                    'type': highlight.get('type')
                })

    df_highlights = pd.DataFrame(highlights_data)

    return fact_covid_features.reset_index(drop=True), df_highlights



df = read_json_from_minio(COVID_FILE)
print(df['highlights'].head(100))
a, b = transform_covid_features(df)
print(b)
