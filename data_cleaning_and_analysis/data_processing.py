import pandas as pd
import boto3
from io import StringIO, BytesIO
import logging
import re
import io

RAW_BUCKET_NAME = "it457-movie-raw-data-mvfy-project-a"       
PROCESSED_BUCKET_NAME = "it457-movie-processed-data-mvfy-project-a" 
REGION = "us-east-1"                                      

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
s3 = boto3.client('s3', region_name=REGION)

def read_csv_from_s3(bucket, key, sep=',', engine='c', names=None):
    """
    Reads a CSV/DAT file from S3 into a pandas DataFrame, 
    handling byte decoding for multi-character delimiters.
    """
    try:
        s3 = boto3.client('s3')
        obj = s3.get_object(Bucket=bucket, Key=key)
        raw_bytes = obj['Body'].read() 
        
        text_stream = io.TextIOWrapper(io.BytesIO(raw_bytes), encoding='latin-1') 
        
        df = pd.read_csv(text_stream, sep=sep, engine=engine, names=names)
        logging.info(f"Successfully read s3://{bucket}/{key}")
        return df
    except Exception as e:
        logging.error(f"Error reading s3://{bucket}/{key}: {e}")
        raise

def write_df_to_s3(df, bucket, key):
    """Writes a pandas DataFrame to S3 as a CSV file."""
    try:
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        s3.put_object(Bucket=bucket, Key=key, Body=csv_buffer.getvalue())
        logging.info(f"Successfully wrote DataFrame to s3://{bucket}/{key}")
    except Exception as e:
        logging.error(f"Error writing DataFrame to s3://{bucket}/{key}: {e}")
        raise

def clean_title(title):
    """Removes the year in parentheses from the title for cleaning."""
    return re.sub(r'\s*\(\d{4}\)$', '', title).strip()

def process_movie_data():
    logging.info("Starting MovieLens 1M data processing...")

    # 1. Load Data - Uses '::' delimiter and manual column names for the DAT files.
    logging.info("Loading raw .dat files from S3...")
    
    # 1.1 Ratings Data
    ratings_cols = ['userId', 'movieId', 'rating', 'timestamp']
    ratings_df = read_csv_from_s3(RAW_BUCKET_NAME, 'raw/ratings.dat', 
                                  sep='::', engine='python', names=ratings_cols)

    # 1.2 Movies Data
    movies_cols = ['movieId', 'title', 'genres']
    movies_df = read_csv_from_s3(RAW_BUCKET_NAME, 'raw/movies.dat', 
                                 sep='::', engine='python', names=movies_cols)

    # 2. Initial Cleaning and Feature Engineering 
    logging.info(f"Loaded {len(ratings_df)} ratings. Cleaning data...")
    
    ratings_df = ratings_df.drop(columns=['timestamp'])
    movies_df['clean_title'] = movies_df['title'].apply(clean_title)

    # 3. Merge Data
    logging.info("Merging ratings and movies data...")
    merged_df = pd.merge(ratings_df, movies_df, on='movieId', how='inner')

    # 4. Final Clean-up and Feature Prep
    merged_df['genres'] = merged_df['genres'].str.replace('|', ' ', regex=False)
    merged_df = merged_df.drop(columns=['title'])

    # 5. Save Processed Data to S3 (This is the input for ML and EDA)
    logging.info("Saving processed 1M data to S3...")
    write_df_to_s3(merged_df, PROCESSED_BUCKET_NAME, 'processed/clean_movie_data_1M.csv')

    logging.info("Data processing complete.")
    return merged_df

if __name__ == '__main__':
    processed_data = process_movie_data()
    logging.info(f"Processed data shape: {processed_data.shape}")
