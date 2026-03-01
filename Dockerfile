FROM python:3.10-slim-buster

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

# Use gunicorn with long timeout for slow model loading
CMD gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 900 --preload app:app
