import math
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

RATINGS_CSV = "clean_movie_data.csv"
RANDOM_STATE = 42

df = pd.read_csv(RATINGS_CSV)

user_ids = df["userId"].unique()
movie_ids = df["movieId"].unique()

user_to_idx = {u: i for i, u in enumerate(user_ids)}
movie_to_idx = {m: i for i, m in enumerate(movie_ids)}

df["user_idx"] = df["userId"].map(user_to_idx)
df["movie_idx"] = df["movieId"].map(movie_to_idx)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)

u_train = train_df["user_idx"].values
i_train = train_df["movie_idx"].values
r_train = train_df["rating"].values.astype(np.float32)

u_test = test_df["user_idx"].values
i_test = test_df["movie_idx"].values
r_test = test_df["rating"].values.astype(np.float32)

PARAM_GRID = [
    {"factors": 80, "lr": 0.004, "reg": 0.015},
    {"factors": 100, "lr": 0.005, "reg": 0.015},
    {"factors": 120, "lr": 0.0045, "reg": 0.012},
]

def train_one(factors, lr, reg, epochs=15):
    n_users = len(user_ids)
    n_items = len(movie_ids)

    rng = np.random.RandomState(42)
    mu = r_train.mean()
    bu = np.zeros(n_users)
    bi = np.zeros(n_items)
    P = 0.01 * rng.randn(n_users, factors)
    Q = 0.01 * rng.randn(n_items, factors)

    indices = np.arange(len(r_train))

    for epoch in range(epochs):
        rng.shuffle(indices)
        for idx in indices:
            u = u_train[idx]
            i = i_train[idx]
            r = r_train[idx]

            pred = mu + bu[u] + bi[i] + np.dot(P[u], Q[i])
            err = r - pred

            # confidence weighting
            conf = 1.0 + 0.2 * r  # boost strong ratings

            bu[u] += lr * (conf * err - reg * bu[u])
            bi[i] += lr * (conf * err - reg * bi[i])

            P[u] += lr * (conf * err * Q[i] - reg * P[u])
            Q[i] += lr * (conf * err * P[u] - reg * Q[i])

    preds = []
    for u, i in zip(u_test, i_test):
        pred = mu + bu[u] + bi[i] + np.dot(P[u], Q[i])
        preds.append(pred)

    rmse = math.sqrt(mean_squared_error(r_test, preds))
    return rmse, (mu, bu, bi, P, Q)

results = []

print("Starting tuning...")

for config in PARAM_GRID:
    print(f"\nTesting config: {config}")
    rmse, model = train_one(config["factors"], config["lr"], config["reg"])
    print(f"RMSE: {rmse:.4f}")
    results.append((rmse, config, model))

best_rmse, best_config, best_model = sorted(results, key=lambda x: x[0])[0]

print("\n===== BEST CONFIG =====")
print(best_config)
print("RMSE:", best_rmse)

# Save best model
mu, bu, bi, P, Q = best_model


# Add index→ID mappings (required for service)
idx_to_movie = {i: m for m, i in movie_to_idx.items()}
idx_to_user = {i: u for u, i in user_to_idx.items()}

artifacts = {
    "mu": mu,
    "bu": bu,
    "bi": bi,
    "P": P,
    "Q": Q,
    "user_to_idx": user_to_idx,
    "movie_to_idx": movie_to_idx,
    "idx_to_movie": idx_to_movie,
    "idx_to_user": idx_to_user,
    "movie_map": pd.read_csv("movie_mapping.csv"),
    "n_factors": best_config["factors"],
}

with open("recommender_artifacts/mf_tuned.pkl", "wb") as f:
    pickle.dump(artifacts, f)

print("\nSaved tuned model with full metadata to recommender_artifacts/mf_tuned.pkl")
