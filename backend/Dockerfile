# Backend Dockerfile
FROM python:3.11-slim

# System deps (optional; Pillow usually works without these on slim)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     libglib2.0-0 libsm6 libxrender1 libxext6 \
#  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy source code
COPY backend /app/backend
COPY data /app/data
COPY plateReader.py /app/plateReader.py

# Expose FastAPI port
EXPOSE 8000

# Environment (can be overridden at runtime)
ENV OLLAMA_API_URL="http://ollama:11434/api/generate"

# Start server
CMD ["python", "-m", "uvicorn", "backend.backend:app", "--host", "0.0.0.0", "--port", "8000"]
