import pandas as pd
import boto3
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import logging
import io

PROCESSED_BUCKET_NAME = "it457-movie-processed-data-mvfy-project-a" 
REGION = "us-east-1"                                                
OUTPUT_PLOT_S3_PATH = "analysis_plots/"                             # S3 folder for plots

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
s3 = boto3.client('s3', region_name=REGION)

def read_df_from_s3(bucket, key):
    """Reads a pandas DataFrame (CSV) from S3."""
    try:
        s3 = boto3.client('s3')
        obj = s3.get_object(Bucket=bucket, Key=key)
        df = pd.read_csv(obj['Body'])
        logging.info(f"Successfully read DataFrame from s3://{bucket}/{key}")
        return df
    except Exception as e:
        logging.error(f"Error reading s3://{bucket}/{key}: {e}")
        raise

def save_plot_to_s3(fig, bucket, key):
    """Saves a matplotlib figure to S3 as a PNG file."""
    try:
        buffer = BytesIO()
        fig.savefig(buffer, format='png')
        buffer.seek(0)
        
        s3.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue(), ContentType='image/png')
        logging.info(f"Successfully saved plot to s3://{bucket}/{key}")
    except Exception as e:
        logging.error(f"Error saving plot to s3://{bucket}/{key}: {e}")
        raise

def perform_eda_and_viz():
    logging.info("Starting EDA and Visualization...")
    sns.set_style("whitegrid")

    # Load processed data
    df = read_df_from_s3(PROCESSED_BUCKET_NAME, 'processed/clean_movie_data_1M.csv')
    
    sample_fraction = 0.10 
    df = df.sample(frac=sample_fraction, random_state=42)
    logging.info(f"Successfully sampled data down to {len(df)} rows ({sample_fraction*100}%).")

    # --- 1. Rating Distribution ---
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(x='rating', data=df, palette='viridis', ax=ax)
    ax.set_title('Distribution of Movie Ratings (All Users)')
    ax.set_xlabel('Rating')
    ax.set_ylabel('Count')
    save_plot_to_s3(fig, PROCESSED_BUCKET_NAME, f"{OUTPUT_PLOT_S3_PATH}rating_distribution.png")
    plt.close(fig)

    # --- 2. Average Rating per Genre ---
    genres_exploded = df.assign(genre=df['genres'].str.split(' ')).explode('genre')
    genre_avg_ratings = genres_exploded.groupby('genre')['rating'].mean().sort_values(ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(x=genre_avg_ratings.index, y=genre_avg_ratings.values, palette='magma', ax=ax)
    ax.set_title('Top 15 Genres by Average Rating')
    ax.set_xlabel('Genre')
    ax.set_ylabel('Average Rating')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    save_plot_to_s3(fig, PROCESSED_BUCKET_NAME, f"{OUTPUT_PLOT_S3_PATH}avg_rating_per_genre.png")
    plt.close(fig)

    # --- 3. Most Rated Movies (Popularity) ---
    most_rated_movies = df['clean_title'].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=most_rated_movies.values, y=most_rated_movies.index, palette='rocket', ax=ax)
    ax.set_title('Top 10 Most Rated Movies (Popularity)')
    ax.set_xlabel('Number of Ratings')
    ax.set_ylabel('Movie Title')
    plt.tight_layout()
    save_plot_to_s3(fig, PROCESSED_BUCKET_NAME, f"{OUTPUT_PLOT_S3_PATH}most_rated_movies.png")
    plt.close(fig)

    # --- 4. Top Rated Movies (Quality, filtered by minimum ratings) ---
    min_ratings_threshold = 50 
    
    movie_stats = df.groupby('clean_title').agg(
        mean_rating=('rating', 'mean'),
        num_ratings=('rating', 'count')
    ).reset_index()
    
    popular_movies = movie_stats[movie_stats['num_ratings'] >= min_ratings_threshold]
    top_rated_popular_movies = popular_movies.sort_values(by='mean_rating', ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=top_rated_popular_movies['mean_rating'], y=top_rated_popular_movies['clean_title'], palette='cubehelix', ax=ax)
    ax.set_title(f'Top 10 Rated Movies (Min {min_ratings_threshold} Ratings)')
    ax.set_xlabel('Average Rating')
    ax.set_ylabel('Movie Title')
    plt.tight_layout()
    save_plot_to_s3(fig, PROCESSED_BUCKET_NAME, f"{OUTPUT_PLOT_S3_PATH}top_rated_movies.png")
    plt.close(fig)

    logging.info("EDA and Visualization complete. Plots saved to S3.")

if __name__ == '__main__':
    perform_eda_and_viz()
