# mf_service_api.py
# FastAPI service for movie recommendations using tuned MF + genres hybrid.

import pickle
import difflib
import re
from typing import List

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.preprocessing import MultiLabelBinarizer

# --------------------- CONFIG ---------------------
RATINGS_CSV = "clean_movie_data.csv"
MODEL_PKL = "recommender_artifacts/mf_tuned.pkl"
MOVIE_MAP_CSV = "movie_mapping.csv"

# --------------------- LOAD DATA & MODEL ON STARTUP ---------------------
print("Loading data and model...")

ratings_df = pd.read_csv(RATINGS_CSV)
movie_map = pd.read_csv(MOVIE_MAP_CSV)

with open(MODEL_PKL, "rb") as f:
    artifacts = pickle.load(f)

mu = artifacts["mu"]
bu = artifacts["bu"]
bi = artifacts["bi"]
P = artifacts["P"]
Q = artifacts["Q"]
user_to_idx = artifacts["user_to_idx"]
movie_to_idx = artifacts["movie_to_idx"]
idx_to_movie = artifacts["idx_to_movie"]

movie_map_indexed = movie_map.set_index("movieId")

# --------------------- GENRE FEATURES ---------------------
movie_genres = ratings_df.groupby("movieId")["genres"].first().reset_index()
movie_genres = movie_map[["movieId"]].merge(movie_genres, on="movieId", how="left")
movie_genres["genres"] = movie_genres["genres"].fillna("")


def parse_genres_str(s):
    if pd.isna(s) or s == "":
        return []
    s2 = str(s).replace("|", " ")
    parts = [p.strip() for p in s2.split() if p.strip() != ""]
    return parts


movie_genres["genres_list"] = movie_genres["genres"].apply(parse_genres_str)

mlb = MultiLabelBinarizer(sparse_output=False)
genre_matrix = mlb.fit_transform(movie_genres["genres_list"])

n_items = len(movie_to_idx)
n_genres = genre_matrix.shape[1]
item_genre_features = np.zeros((n_items, n_genres), dtype=np.float32)

for i, row in movie_genres.iterrows():
    mid = row["movieId"]
    if mid in movie_to_idx:
        idx = movie_to_idx[mid]
        item_genre_features[idx, :] = genre_matrix[i]

# --------------------- TITLE NORMALIZATION & RESOLUTION ---------------------
def normalize_title(t: str) -> str:
    t = t.lower().strip()
    t = re.sub(r"[^a-z0-9 ]", "", t)  # remove punctuation
    return " ".join(t.split())        # collapse extra spaces


normalized_title_to_id = {}
for mid, row in movie_map_indexed.iterrows():
    raw = str(row["clean_title"])
    norm = normalize_title(raw)
    normalized_title_to_id[norm] = mid


def resolve_titles_to_ids(movie_titles: List[str]):
    resolved = []
    missing = []

    for title in movie_titles:
        norm = normalize_title(title)

        # 1) exact normalized match
        if norm in normalized_title_to_id:
            resolved.append(normalized_title_to_id[norm])
            continue

        # 2) reverse-word format fix (e.g., "the matrix" vs "matrix the")
        parts = norm.split()
        if len(parts) > 1:
            rev = " ".join(parts[1:] + [parts[0]])
            if rev in normalized_title_to_id:
                resolved.append(normalized_title_to_id[rev])
                continue

        # 3) fuzzy match fallback
        candidates = difflib.get_close_matches(
            norm, normalized_title_to_id.keys(), n=1, cutoff=0.45
        )
        if candidates:
            resolved.append(normalized_title_to_id[candidates[0]])
        else:
            missing.append(title)

    return resolved, missing

# --------------------- CORE MF FUNCTIONS ---------------------
def predict_rating_mf(user_id: int, movie_id: int) -> float:
    """
    Predict rating for a known user + movie pair using MF.
    """
    if user_id not in user_to_idx or movie_id not in movie_to_idx:
        return float(mu)

    u = user_to_idx[user_id]
    i = movie_to_idx[movie_id]
    pu = P[u]
    qi = Q[i]
    pred = mu + bu[u] + bi[i] + float(np.dot(pu, qi))
    return float(pred)


def recommend_from_watched(movie_list, n: int = 10, alpha: float = 0.7):
    """
    Recommend movies based on a list of movieIds the user already watched/liked.
    No user ID needed.

    final_score = alpha * MF_score_norm + (1 - alpha) * genre_sim_norm
    """

    valid_movies = [m for m in movie_list if m in movie_to_idx]
    if not valid_movies:
        return []

    # pseudo-user latent vector (MF side)
    q_vectors = [Q[movie_to_idx[m]] for m in valid_movies]
    pseudo_user = np.mean(q_vectors, axis=0)

    # pseudo-user genre profile (content side)
    g_vectors = [item_genre_features[movie_to_idx[m]] for m in valid_movies]
    genre_profile = np.mean(g_vectors, axis=0)
    genre_profile_norm = np.linalg.norm(genre_profile)

    mf_scores = []
    sim_scores = []
    mids = []

    for mid, idx in movie_to_idx.items():
        if mid in valid_movies:
            continue  # skip already watched

        qi = Q[idx]
        mf = mu + bi[idx] + float(np.dot(pseudo_user, qi))

        gi = item_genre_features[idx]
        gi_norm = np.linalg.norm(gi)
        if genre_profile_norm > 0 and gi_norm > 0:
            sim = float(np.dot(gi, genre_profile) / (genre_profile_norm * gi_norm))
        else:
            sim = 0.0

        mids.append(mid)
        mf_scores.append(mf)
        sim_scores.append(sim)

    mf_scores = np.array(mf_scores, dtype=np.float32)
    sim_scores = np.array(sim_scores, dtype=np.float32)

    # normalize
    mf_mean, mf_std = mf_scores.mean(), mf_scores.std()
    if mf_std > 1e-8:
        mf_norm = (mf_scores - mf_mean) / mf_std
    else:
        mf_norm = mf_scores - mf_mean

    sim_mean, sim_std = sim_scores.mean(), sim_scores.std()
    if sim_std > 1e-8:
        sim_norm = (sim_scores - sim_mean) / sim_std
    else:
        sim_norm = sim_scores - sim_mean

    final_scores = alpha * mf_norm + (1.0 - alpha) * sim_norm

    idx_sorted = np.argsort(final_scores)[::-1]
    top_idx = idx_sorted[:n]

    results = []
    for k in top_idx:
        mid = mids[k]
        s = final_scores[k]
        title = (
            movie_map_indexed.loc[mid]["clean_title"]
            if mid in movie_map_indexed.index
            else "UNKNOWN"
        )
        results.append(
            {
                "movieId": int(mid),
                "title": str(title),
                "score": float(s),
            }
        )

    return results


def recommend_from_titles(movie_titles: List[str], n: int = 10, alpha: float = 0.7):
    """
    Same as recommend_from_watched, but takes movie names instead of IDs.
    Returns (recommendations, missing_titles)
    """
    movie_ids, missing = resolve_titles_to_ids(movie_titles)

    if not movie_ids:
        return [], missing

    recs = recommend_from_watched(movie_ids, n=n, alpha=alpha)
    return recs, missing

# --------------------- FASTAPI SCHEMA ---------------------
class RecommendTitlesRequest(BaseModel):
    movies: List[str]
    top_k: int = 10
    alpha: float = 0.7


class RecommendationItem(BaseModel):
    movieId: int
    title: str
    score: float


class RecommendTitlesResponse(BaseModel):
    input_movies: List[str]
    missing_movies: List[str]
    recommendations: List[RecommendationItem]


class PredictRatingRequest(BaseModel):
    user_id: int
    movie_id: int


class PredictRatingResponse(BaseModel):
    user_id: int
    movie_id: int
    predicted_rating: float


# --------------------- FASTAPI APP ---------------------
app = FastAPI(
    title="Cloud Movie Recommender API",
    version="1.0.0",
    description="Matrix Factorization + Genre Hybrid Recommender",
)

# CORS so frontend (http://127.0.0.1:5500 etc.) can call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # for dev; you can restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/recommend/titles", response_model=RecommendTitlesResponse)
def recommend_by_titles(payload: RecommendTitlesRequest):
    recs, missing = recommend_from_titles(
        payload.movies, n=payload.top_k, alpha=payload.alpha
    )

    rec_items = [RecommendationItem(**r) for r in recs]

    return RecommendTitlesResponse(
        input_movies=payload.movies,
        missing_movies=missing,
        recommendations=rec_items,
    )


@app.post("/predict-rating", response_model=PredictRatingResponse)
def predict_rating(payload: PredictRatingRequest):
    pred = predict_rating_mf(payload.user_id, payload.movie_id)
    return PredictRatingResponse(
        user_id=payload.user_id,
        movie_id=payload.movie_id,
        predicted_rating=pred,
    )
