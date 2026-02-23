# Backend Dockerfile
FROM python:3.11-slim



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

ENV OLLAMA_API_URL="http://ollama:11434/api/generate"

# Start server
CMD ["python", "-m", "uvicorn", "backend.backend:app", "--host", "0.0.0.0", "--port", "8000"]
