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

def create_dim_datetime(start_date='1948-09-06', end_date='2025-12-31'):
    """
    Generates a datetime dimension table with daily granularity.
    For hour, minute, second level, we use a separate hour dimension table.
    """
    # Create date range with daily frequency instead of seconds
    date_rng = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Create base dataframe with just the dates
    dim_datetime = pd.DataFrame(date_rng, columns=['full_timestamp'])
    dim_datetime['datetime_id'] = range(len(dim_datetime))
    dim_datetime['date_id'] = dim_datetime['full_timestamp'].dt.strftime('%Y%m%d').astype(int)
    
    # Add time components
    dim_datetime['day_of_week'] = dim_datetime['full_timestamp'].dt.day_name()
    dim_datetime['day_of_month'] = dim_datetime['full_timestamp'].dt.day
    dim_datetime['month'] = dim_datetime['full_timestamp'].dt.month
    dim_datetime['year'] = dim_datetime['full_timestamp'].dt.year
    dim_datetime['quarter'] = dim_datetime['full_timestamp'].dt.quarter
    
    # Create a mapping for each hour of the day (0-23)
    hours = range(24)
    hour_mapping = {hour: hour for hour in hours}
    
    # Add hour mapping to the dimension table
    dim_datetime['hour_id'] = 0  # Default value
    
    return dim_datetime

def create_dim_hour():
    """Create hour dimension table."""
    hours = range(24)
    dim_hour = pd.DataFrame({
        'hour_id': hours, 
        'hour_of_day': [f'{h:02d}' for h in hours]
    })
    return dim_hour

def create_dim_date(dim_datetime_df):
    """Create date dimension from datetime dimension."""
    dim_date_df = dim_datetime_df[['date_id', 'full_timestamp']].drop_duplicates(subset=['date_id'])
    dim_date_df['full_date'] = dim_date_df['full_timestamp'].dt.date
    dim_date_df['year'] = dim_date_df['full_timestamp'].dt.year
    dim_date_df['month'] = dim_date_df['full_timestamp'].dt.month
    dim_date_df['day'] = dim_date_df['full_timestamp'].dt.day
    dim_date_df['day_of_week'] = dim_date_df['full_timestamp'].dt.day_name()
    dim_date_df['quarter'] = dim_date_df['full_timestamp'].dt.quarter
    dim_date_df.drop(columns=['full_timestamp'], inplace=True)
    return dim_date_df.reset_index(drop=True)

# --- Transformation Functions ---
def transform_business(df, dim_hour_df):
    """Transform business data into dimensional model."""
    # Extract business dimension
    dim_business = df[['business_id', 'name', 'address', 'city', 'state', 
                       'postal_code', 'latitude', 'longitude', 'is_open', 
                       'stars', 'review_count']].drop_duplicates()
    
    # Process categories
    # Use more efficient method to explode categories
    categories_data = []
    for _, row in df[['business_id', 'categories']].iterrows():
        if pd.notna(row['categories']) and row['categories']:
            for category in row['categories'].split(', '):
                categories_data.append({'business_id': row['business_id'], 'category_name': category})
    
    df_categories = pd.DataFrame(categories_data)
    
    # Create category dimension
    if not df_categories.empty:
        dim_categories = df_categories[['category_name']].drop_duplicates().dropna()
        dim_categories['category_id'] = np.arange(1, len(dim_categories) + 1)
        
        # Create business-category fact table
        fact_business_categories = pd.merge(
            df_categories,
            dim_categories,
            on='category_name',
            how='inner'
        )[['business_id', 'category_id']]
    else:
        dim_categories = pd.DataFrame(columns=['category_name', 'category_id'])
        fact_business_categories = pd.DataFrame(columns=['business_id', 'category_id'])
    
    # Process attributes
    attributes_data = []
    for _, row in df[['business_id', 'attributes']].iterrows():
        if isinstance(row['attributes'], dict):
            for key, value in row['attributes'].items():
                attributes_data.append({
                    'business_id': row['business_id'], 
                    'attribute_name': key, 
                    'attribute_value': str(value)
                })
    
    # Create attribute dimension
    if attributes_data:
        attributes_df = pd.DataFrame(attributes_data)
        dim_attributes = attributes_df[['attribute_name', 'attribute_value']].drop_duplicates()
        dim_attributes['attribute_id'] = np.arange(1, len(dim_attributes) + 1)
        
        # Create business-attribute fact table
        fact_business_attributes = pd.merge(
            attributes_df,
            dim_attributes,
            on=['attribute_name', 'attribute_value'],
            how='inner'
        )[['business_id', 'attribute_id']]
    else:
        dim_attributes = pd.DataFrame(columns=['attribute_name', 'attribute_value', 'attribute_id'])
        fact_business_attributes = pd.DataFrame(columns=['business_id', 'attribute_id'])
    
    # Process hours
    hours_data = []
    for _, row in df[['business_id', 'hours']].iterrows():
        if isinstance(row['hours'], dict):
            for day, time_range in row['hours'].items():
                if '-' in time_range:
                    open_time_str, close_time_str = time_range.split('-')
                    try:
                        open_hour = int(open_time_str.split(':')[0])
                        close_hour = int(close_time_str.split(':')[0]) % 24
                        
                        # Use vectorized operations for lookup
                        open_hour_id = open_hour  # Direct mapping
                        close_hour_id = close_hour  # Direct mapping
                        
                        hours_data.append({
                            'business_id': row['business_id'],
                            'day_of_week': day,
                            'open_time': open_time_str,
                            'close_time': close_time_str,
                            'open_hour_id': open_hour_id,
                            'close_hour_id': close_hour_id
                        })
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Could not parse hours for business {row['business_id']} on {day}: {time_range}. Error: {e}")
    
    fact_business_hours = pd.DataFrame(hours_data) if hours_data else pd.DataFrame(
        columns=['business_id', 'day_of_week', 'open_time', 'close_time', 'open_hour_id', 'close_hour_id']
    )
    
    return (
        dim_business.reset_index(drop=True),
        dim_categories.reset_index(drop=True),
        fact_business_categories.reset_index(drop=True),
        dim_attributes.reset_index(drop=True),
        fact_business_attributes.reset_index(drop=True),
        fact_business_hours.reset_index(drop=True)
    )

def transform_checkin(df, dim_datetime_df):
    """Transform check-in data into fact table."""
    if df.empty:
        return pd.DataFrame(columns=['checkin_id', 'business_id', 'datetime_id'])
    
    # Optimized check-in processing using explode and vectorized operations
    logger.info("Optimizing checkin transformation...")

    # Create a datetime lookup series for faster mapping
    datetime_lookup = dim_datetime_df.set_index(dim_datetime_df['full_timestamp'].dt.strftime('%Y-%m-%d'))['datetime_id']

    # 1. Split the 'date' string into a list of dates
    df['date_list'] = df['date'].str.split(',')

    # 2. Explode the dataframe based on the list of dates
    #    This creates a new row for each date in the list
    exploded_df = df.explode('date_list')

    # 3. Clean up the date string and convert to datetime, then format
    #    Handle potential errors during conversion by setting errors='coerce'
    exploded_df['date_str'] = pd.to_datetime(exploded_df['date_list'].str.strip(), errors='coerce').dt.strftime('%Y-%m-%d')

    # 4. Drop rows where date conversion failed or date is missing
    exploded_df.dropna(subset=['date_str'], inplace=True)

    # 5. Map the formatted date string to datetime_id using the lookup series
    exploded_df['datetime_id'] = exploded_df['date_str'].map(datetime_lookup)

    # 6. Drop rows where the date didn't match anything in dim_datetime
    exploded_df.dropna(subset=['datetime_id'], inplace=True)

    # 7. Select and rename final columns
    fact_checkins = exploded_df[['business_id', 'datetime_id']].copy()
    fact_checkins['datetime_id'] = fact_checkins['datetime_id'].astype(int) # Ensure correct type

    # 8. Add checkin_id
    fact_checkins.reset_index(drop=True, inplace=True)
    fact_checkins['checkin_id'] = np.arange(1, len(fact_checkins) + 1)

    logger.info(f"Finished optimizing checkin transformation. Processed {len(fact_checkins)} checkins.")
    return fact_checkins

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

def transform_review(df, dim_datetime_df):
    """Transform review data into fact table."""
    if df.empty:
        return pd.DataFrame(columns=[
            'review_id', 'business_id', 'user_id', 'stars', 
            'useful', 'funny', 'cool', 'text', 'datetime_id'
        ])
    
    # Create a datetime lookup dictionary for faster lookups
    datetime_lookup = dict(zip(
        dim_datetime_df['full_timestamp'].dt.strftime('%Y%m%d'),
        dim_datetime_df['datetime_id']
    ))

    # Process in chunks for large datasets
    chunks = []
    for chunk_df in np.array_split(df, max(1, len(df) // CHUNK_SIZE)):
        # Extract needed columns
        chunk_result = chunk_df[['review_id', 'business_id', 'user_id', 
                                'stars', 'useful', 'funny', 'cool', 'text', 'date']].copy()
        
        # Convert dates to datetime and extract date part
        chunk_result['date_str'] = pd.to_datetime(chunk_result['date']).dt.strftime('%Y-%m-%d')
        
        # Map to datetime_id using our lookup
        chunk_result['datetime_id'] = chunk_result['date_str'].map(datetime_lookup)
        
        # Drop unnecessary columns
        chunk_result.drop(columns=['date', 'date_str'], inplace=True)
        
        chunks.append(chunk_result)
    
    if not chunks:
        return pd.DataFrame(columns=[
            'review_id', 'business_id', 'user_id', 'stars', 
            'useful', 'funny', 'cool', 'text', 'datetime_id'
        ])
    
    return pd.concat(chunks, ignore_index=True)

def transform_user(df):
    """Transform user data into dimensional model."""
    if df.empty:
        return (
            pd.DataFrame(columns=['user_id', 'name', 'review_count', 'yelping_since', 'useful', 'funny', 'cool', 'fans', 'average_stars', 'compliment_hot', 'compliment_more', 'compliment_profile', 'compliment_cute', 'compliment_list', 'compliment_note', 'compliment_plain', 'compliment_cool', 'compliment_funny', 'compliment_writer', 'compliment_photos']),
            pd.DataFrame(columns=['elite_id', 'elite_year']),
            pd.DataFrame(columns=['friend_id', 'friend_name']), # Modified
            pd.DataFrame(columns=['user_id', 'elite_id']),
            pd.DataFrame(columns=['user_id', 'friend_id']) # Modified
        )

    # Extract needed columns
    dim_user = df[['user_id', 'name', 'review_count', 'yelping_since',
                  'useful', 'funny', 'cool', 'fans', 'average_stars',
                  'compliment_hot', 'compliment_more', 'compliment_profile',
                  'compliment_cute', 'compliment_list', 'compliment_note',
                  'compliment_plain', 'compliment_cool', 'compliment_funny',
                  'compliment_writer', 'compliment_photos']].copy()

    # Convert yelping_since to datetime
    dim_user['yelping_since'] = pd.to_datetime(dim_user['yelping_since'])

    # Process elite years
    df_elite = df[['user_id', 'elite']].copy()
    df_elite['elite'] = df_elite['elite'].str.split(',')
    df_elite = df_elite.explode('elite')
    df_elite.rename(columns={'elite': 'elite_year'}, inplace=True)
    df_elite = df_elite[df_elite['elite_year'].notna() & (df_elite['elite_year'] != '')]
    dim_elite = df_elite[['elite_year']].drop_duplicates().reset_index(drop=True)
    dim_elite['elite_id'] = dim_elite.index + 1
    fact_user_elite = pd.merge(df_elite, dim_elite, on='elite_year', how='left')[['user_id', 'elite_id']]

    # Process friends
    df_friends = df[['user_id', 'friends']].copy()
    df_friends['friends'] = df_friends['friends'].str.split(',')
    df_friends = df_friends.explode('friends')
    df_friends.rename(columns={'friends': 'friends_name'}, inplace=True)
    df_friends = df_friends[df_friends['friends_name'].notna() & (df_friends['friends_name'] != '')]
    dim_friend = df_friends[['friends_name']].drop_duplicates().reset_index(drop=True)
    dim_friend['friend_id'] = dim_friend.index+1
    fact_user_friend = pd.merge(df_friends, dim_friend, on='friends_name', how='left')[['user_id', 'friend_id']]

    return (
        dim_user.reset_index(drop=True),
        dim_elite.reset_index(drop=True),
        dim_friend.reset_index(drop=True), # Modified
        fact_user_elite.reset_index(drop=True),
        fact_user_friend.reset_index(drop=True) # Modified
    )

def transform_tip(df, dim_datetime_df):
    """Transform tip data into fact table."""
    if df.empty:
        return pd.DataFrame(columns=[
            'tip_id', 'text', 'compliment_count', 'business_id', 'user_id', 'datetime_id'
        ])

    # Create a datetime lookup dictionary for faster lookups
    datetime_lookup = dict(zip(
        dim_datetime_df['full_timestamp'].dt.strftime('%Y-%m-%d'),
        dim_datetime_df['datetime_id']
    ))

    # Extract needed columns
    fact_tips = df[['text', 'compliment_count', 'business_id', 'user_id', 'date']].copy()

    # Convert dates to datetime and extract date part
    fact_tips['date_str'] = pd.to_datetime(fact_tips['date']).dt.strftime('%Y-%m-%d')

    # Map to datetime_id using our lookup
    fact_tips['datetime_id'] = fact_tips['date_str'].map(datetime_lookup)

    # Drop unnecessary columns
    fact_tips.drop(columns=['date', 'date_str'], inplace=True)

    # Add tip_id
    fact_tips['tip_id'] = np.arange(1, len(fact_tips) + 1)

    return fact_tips.reset_index(drop=True)

def transform_temperature(df, dim_datetime_df):
    """Transform temperature data into dimension table."""
    if df.empty:
        return pd.DataFrame(columns=[
            'datetime_id', 'min_temperature', 'max_temperature', 
            'normal_min_temperature', 'normal_max_temperature'
        ])
    df['date'] = df['date'].astype(str)
    # Create a datetime lookup dictionary for faster lookups
    datetime_lookup = dict(zip(
        dim_datetime_df['full_timestamp'].dt.strftime('%Y%m%d'),
        dim_datetime_df['datetime_id']
    ))

    # Rename columns
    dim_temperature = df.rename(columns={
        'date': 'date_str',
        'min': 'min_temperature',
        'max': 'max_temperature',
        'normal_min': 'normal_min_temperature',
        'normal_max': 'normal_max_temperature'
    })

    # Map to datetime_id using our lookup
    dim_temperature['datetime_id'] = dim_temperature['date_str'].map(datetime_lookup)

    # Set index
    dim_temperature = dim_temperature.set_index('datetime_id')

    # Drop unnecessary columns
    dim_temperature.drop(columns=['date_str'], inplace=True)

    # Reset index to make datetime_id a column again
    dim_temperature = dim_temperature.reset_index()

    return dim_temperature

def transform_precipitation(df, dim_datetime_df):
    """Transform precipitation data into dimension table."""
    if df.empty:
        return pd.DataFrame(columns=[
            'datetime_id', 'precipitation', 'normal_precipitation'
        ])
    df['date'] = df['date'].astype(str)
    # Create a datetime lookup dictionary for faster lookups
    datetime_lookup = dict(zip(
        dim_datetime_df['full_timestamp'].dt.strftime('%Y%m%d'),
        dim_datetime_df['datetime_id']
    ))

    # Rename columns
    dim_precipitation = df.rename(columns={
        'date': 'date_str',
        'precipitation_normal': 'normal_precipitation'
    })

    # Map to datetime_id using our lookup
    dim_precipitation['datetime_id'] = dim_precipitation['date_str'].map(datetime_lookup)

    # Set index
    dim_precipitation = dim_precipitation.set_index('datetime_id')

    # Drop unnecessary columns
    dim_precipitation.drop(columns=['date_str'], inplace=True)

    # Reset index to make datetime_id a column again
    dim_precipitation = dim_precipitation.reset_index()

    return dim_precipitation

# --- Load to DuckDB ---
def load_to_duckdb(dataframes, duckdb_file=DUCKDB_DB_FILE):
    """Load dataframes to DuckDB database."""
    logger.info(f"Connecting to DuckDB database: {duckdb_file}")
    try:
        con = duckdb.connect(database=duckdb_file, read_only=False)
        logger.info("Successfully connected to DuckDB.")

        # Process each table
        for table_name, df in dataframes.items():
            logger.info(f"Attempting to load table: {table_name}")
            try:
                if df is not None and not df.empty:
                    # Register the DataFrame with DuckDB under a temporary name
                    temp_df_name = f"temp_{table_name}_df"
                    con.register(temp_df_name, df)
                    # Create the table by selecting from the registered DataFrame
                    con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM {temp_df_name}")
                    # Unregister the temporary DataFrame to free up memory
                    con.unregister(temp_df_name)
                    logger.info(f"Successfully loaded {table_name} to DuckDB.")
                else:
                    logger.warning(f"Skipped loading empty table {table_name}.")
            except Exception as e:
                logger.error(f"Error loading {table_name} to DuckDB: {e}")

        con.close()
        logger.info(f"DuckDB database saved to {duckdb_file}")
        return True
    except Exception as e:
        logger.error(f"Error loading data to DuckDB: {e}")
        return False

# --- Parallel Processing Functions ---
def process_dataset(dataset_name, dim_datetime_df, dim_hour_df):
    """Process a single dataset based on its name."""
    logger.info(f"Processing {dataset_name} dataset...")
    
    try:
        if dataset_name == 'business':
            df = read_json_from_minio(BUSINESS_FILE)
            if df is not None:
                return transform_business(df, dim_hour_df)
            
        elif dataset_name == 'checkin':
            df = read_json_from_minio(CHECKIN_FILE)
            if df is not None:
                return transform_checkin(df, dim_datetime_df)
            
        elif dataset_name == 'covid':
            df = read_json_from_minio(COVID_FILE)
            if df is not None:
                return transform_covid_features(df)
            
        elif dataset_name == 'review':
            df = read_json_from_minio(REVIEW_FILE)
            if df is not None:
                return transform_review(df, dim_datetime_df)
            
        elif dataset_name == 'user':
            df = read_json_from_minio(USER_FILE)
            if df is not None:
                return transform_user(df)
            
        elif dataset_name == 'tip':
            df = read_json_from_minio(TIP_FILE)
            if df is not None:
                return transform_tip(df, dim_datetime_df)
            
        elif dataset_name == 'temperature':
            df = read_csv_from_minio(TEMPERATURE_FILE)
            if df is not None:
                return transform_temperature(df, dim_datetime_df)

        elif dataset_name == 'precipitation':
            df = read_csv_from_minio(PRECIPITATION_FILE)
            if df is not None:
                return transform_precipitation(df, dim_datetime_df)
        
        logger.warning(f"No data processed for {dataset_name}")
        return None
    except Exception as e:
        logger.error(f"Error processing {dataset_name} dataset: {e}")
        return None

# --- Main ETL Process ---
def main():
    """Main ETL process with parallel processing."""
    logger.info("Starting Yelp Data ETL Pipeline...")
    start_time = datetime.now()
    
    try:
        # Create dimension tables first (these are needed for all transformations)
        logger.info("Creating dimension tables...")
        dim_datetime_df = create_dim_datetime(start_date='1948-09-06', end_date='2025-12-31')
        dim_hour_df = create_dim_hour()
        dim_date_df = create_dim_date(dim_datetime_df)
        
        # Define datasets to process
        datasets = ['business', 'checkin', 'covid', 'review', 'user', 'tip', 'temperature', 'precipitation']
        
        # Process datasets in parallel
        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Create a partial function with fixed arguments
            process_func = partial(process_dataset, dim_datetime_df=dim_datetime_df, dim_hour_df=dim_hour_df)
            
            # Submit all tasks
            future_to_dataset = {executor.submit(process_func, dataset): dataset for dataset in datasets}
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_dataset):
                dataset = future_to_dataset[future]
                try:
                    result = future.result()
                    if result is not None:
                        results[dataset] = result
                        logger.info(f"Successfully processed {dataset} dataset")
                    else:
                        logger.warning(f"No result for {dataset} dataset")
                except Exception as e:
                    logger.error(f"Error processing {dataset} dataset: {e}")

        logger.info("Finished processing all datasets in parallel.") # <-- ADDED LOG

        # Prepare data for loading
        logger.info("Preparing data for DuckDB loading...") # <-- ADDED LOG
        data_to_load = {
            'dim_datetime': dim_datetime_df,
            'dim_date': dim_date_df,
            'dim_hour': dim_hour_df
        }
        
        # Add results from parallel processing
        if 'business' in results:
            dim_business_df, dim_categories_df, fact_business_categories_df, dim_attributes_df, fact_business_attributes_df, fact_business_hours_df = results['business']
            data_to_load.update({
                'dim_business': dim_business_df,
                'dim_category': dim_categories_df,
                'fact_business_categories': fact_business_categories_df,
                'dim_attribute': dim_attributes_df,
                'fact_business_attributes': fact_business_attributes_df,
                'fact_business_hours': fact_business_hours_df
            })
        if 'user' in results:
            dim_user, dim_elite, dim_friend, fact_user_elite, fact_user_friend = results['user']
            data_to_load['dim_user'] = dim_user
            data_to_load['dim_elite'] = dim_elite
            data_to_load['dim_friend'] = dim_friend
            data_to_load['fact_user_elite'] = fact_user_elite
            data_to_load['fact_user_friend'] = fact_user_friend
        
        if 'review' in results:
            data_to_load['fact_reviews'] = results['review']
        
        if 'checkin' in results:
            data_to_load['fact_checkins'] = results['checkin']
        
        if 'tip' in results:
            data_to_load['fact_tips'] = results['tip']
        if 'covid' in results:
            fact_covid_features, dim_highlights = results['covid']
            data_to_load['fact_covid_features'] = fact_covid_features
            data_to_load['dim_highlights'] = dim_highlights
        
        if 'temperature' in results:
            data_to_load['dim_temperature'] = results['temperature']
        
        if 'precipitation' in results:
            data_to_load['dim_precipitation'] = results['precipitation']

        # Load data to DuckDB
        if load_to_duckdb(data_to_load):
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.info(f"Yelp Data ETL Pipeline finished successfully in {duration:.2f} seconds.")
        else:
            logger.error("Failed to load data to DuckDB.")
    
    except Exception as e:
        logger.error(f"Error in ETL pipeline: {e}")
        raise

if __name__ == "__main__":
    main()
