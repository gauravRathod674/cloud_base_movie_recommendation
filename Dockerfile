# Dockerfile for Cloud Movie Recommender API

FROM python:3.11-slim

# Prevents Python from writing .pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies (optional but often useful for pandas / sklearn)
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY mf_service_api.py .
COPY clean_movie_data.csv .
COPY movie_mapping.csv .
COPY recommender_artifacts ./recommender_artifacts

# Expose the port FastAPI will run on
EXPOSE 8000

# Run the app with uvicorn
CMD ["uvicorn", "mf_service_api:app", "--host", "0.0.0.0", "--port", "8000"]
