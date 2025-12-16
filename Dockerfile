# Use official Python slim image
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

# Install system packages needed for OpenCV, ffmpeg etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install first (cache layer)
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app

# Ensure uploads dir exists
RUN mkdir -p /app/static/uploads

# Expose flask port
EXPOSE 5000

# Environment variables (can be overridden at runtime)
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV FLASK_SECRET="change_this_in_prod"

# Use Gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app", "--workers", "1", "--threads", "4", "--timeout", "120"]
