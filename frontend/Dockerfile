# Frontend Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Copy and install dependencies
COPY frontend/requirements.txt /app/frontend/requirements.txt
RUN pip install --no-cache-dir -r /app/frontend/requirements.txt

# Copy source code
COPY frontend /app/frontend

# Expose Gradio port
EXPOSE 7860

# Default backend URL can be overridden at deploy time
ENV BACKEND_URL="http://backend:8000"

CMD ["python", "frontend/main.py"]
