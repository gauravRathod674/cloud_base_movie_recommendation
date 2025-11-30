# 🎬 Cloud-Based Movie Recommendation System (IT457)

A production-ready **cloud deployed movie recommender** built using:

* **Matrix Factorization (MF-SGD) Recommendation Model**
* **Hybrid Filtering (Collaborative + Content-Based)**
* **FastAPI Backend**
* **Beautiful Dark-Theme Interactive Frontend**
* **Docker Deployment + Cloud Ready Architecture**

This project lets users **enter movies they like**, and the system returns **personalized recommendations** in real-time — even without stored user login history.

---

## 🚀 Features

✔ Hybrid Recommendation Engine
✔ Accepts Movie Titles (Fuzzy-Matching Supported)
✔ Real-Time FastAPI Service (JSON Output)
✔ Docker Deployable
✔ Cloud Ready (AWS: S3 + EC2 + API Gateway Ready)
✔ Beautiful Dark UI Frontend
✔ Includes Data Analysis Dashboard (Rating Distribution, Trends, Top Movies)

---

## 🧠 Model Overview

The model is trained using **Matrix Factorization with SGD improvements** and combines:

| Technique                             | Purpose                                                       |
| ------------------------------------- | ------------------------------------------------------------- |
| **Collaborative Filtering (MF-SGD)**  | Learns user/movie latent features                             |
| **Genre-Based Content Similarity**    | Helps when a movie or user has sparse ratings                 |
| **Fuzzy Title Matching System**       | Cleans misspelled user input                                  |
| **Confidence-Based Gradient Updates** | Boost accuracy by treating higher ratings as more influential |

Best tuned configuration:

```json
{
  "factors": 100,
  "learning_rate": 0.005,
  "regularization": 0.015,
  "RMSE": 0.8605
}
```

---

## 🗂️ Project Structure

```
final_project/
│
├── frontend/
│   └── index.html            # Modern UI
│
├── recommender_artifacts/
│   ├── mf_tuned.pkl          # Final trained model
│   └── mf_sgd_model.pkl      # Earlier versions
│
├── clean_movie_data.csv
├── movie_mapping.csv
├── mf_service_api.py         # FastAPI Backend
├── tune_mf.py                # Model Training & Tuning Script
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## ⚙️ Installation (Local)

### 1️⃣ Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run Backend (FastAPI)

```bash
uvicorn mf_service_api:app --reload
```

Backend available at:

👉 `http://127.0.0.1:8000/docs`

### 4️⃣ Run Frontend

Right-click `frontend/index.html` → **Open in browser**
(or use Live Server extension)

---

## 🐳 Deploy Using Docker

### Build Image:

```bash
docker build -t cloud-movie-api .
```

### Run Container:

```bash
docker run --name cloud-movie-api-container -p 8000:8000 cloud-movie-api
```

Verify:

👉 `http://127.0.0.1:8000/docs`

---

## 🌩️ Deploy to AWS (Summary)

| Service                    | Purpose                 |
| -------------------------- | ----------------------- |
| **AWS EC2 OR ECR + ECS**   | Runs the Docker backend |
| **AWS S3 + CloudFront**    | Hosts the frontend      |
| **AWS Route53**            | Domain + SSL            |
| **(Optional) API Gateway** | For public API handling |

(Fully deployable with same Docker image — no code changes required.)

---

## 🧪 API Endpoints

| Method | Endpoint            | Purpose                            |
| ------ | ------------------- | ---------------------------------- |
| `GET`  | `/health`           | Status check                       |
| `POST` | `/recommend/titles` | Generate Movie Recommendations     |
| `POST` | `/predict-rating`   | Predict rating for user–movie pair |

Example Request:

```json
{
  "movies": ["Toy Story", "The Matrix", "Titanic"],
  "top_k": 10,
  "alpha": 0.7
}
```

---

## 🎨 Frontend Preview

✔ Add movies
✔ Remove movies
✔ Live recommendations
✔ Dashboard visuals (Graphs loaded from S3)

---

## 📊 System Architecture

```
User → Frontend (HTML/JS) → FastAPI Backend → Model (.pkl) → Response
                             ↓
                        Hybrid Engine
         MF-SGD + Genre Similarity + Fuzzy Matching
```

---

## 📁 Dataset

* Based on **MovieLens 100K/1M rating dataset**
* Movies mapped & preprocessed
* Cleaned formatting for reliable matching

---

## 🧪 Example Output

```json
{
  "input_movies": ["Toy Story", "The Matrix"],
  "recommendations": [
    { "title": "Toy Story 2", "score": 3.98 },
    { "title": "Terminator 2: Judgment Day", "score": 3.74 }
  ]
}
```

---

## 🏆 Team & Credits

| Role               | Name                    |
| ------------------ | ----------------------- |
| Lead Developer     | **Gaurav Rathod**       |
| Model & Deployment | **Cloud + ML Pipeline** |
| UI / System Design | 💻                      |

---

## ⭐ Future Improvements

* Transformer-based embeddings (BERT/LightFM)
* Real user authentication + watch history tracking
* Review sentiment analysis integration
* Improved UI animations + mobile responsive mode

---

## 📌 License

This project is developed for **IT457: Cloud Computing Course** academic use.

---

