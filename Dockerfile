FROM python:3.10-slim-buster

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

# Preload the embedding model during build (faster startup)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Use gunicorn with long timeout for slow model loading
CMD gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 300 --preload app:app
