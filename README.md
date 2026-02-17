# Tunisian License Plate Reader

A full-stack application for detecting and reading Tunisian license plates using Roboflow for detection, PaddleOCR for character recognition, and Supabase for driver information lookup.

## Project Structure

- `backend/`: FastAPI server for plate detection and OCR.
- `frontend/`: Gradio-based user interface.
- `data/`: Contains training images, test images, and database CSVs.
- `plateReader.py`: Script for standalone detection testing.
- `upload_data.py`: Script to upload images and annotations to Roboflow.
- `requirements.txt`: Project dependencies.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r frontend/requirements.txt
   ```
2. Configure `.env` file with your API keys:
   - `RF_API_KEY`: Roboflow API key
   - `SUPABASE_URL`: Supabase project URL
   - `SUPABASE_ANON_KEY`: Supabase anonymous key
   - `OCR_API_URL`: PaddleOCR endpoint URL
   - `OCR_TOKEN`: PaddleOCR access token

3. (Optional) Install and start **Ollama** for the AI Assistant:
   - Download Ollama from [ollama.com](https://ollama.com).
   - Pull the required model: `ollama pull llama3.2:3b`.
   - Ensure Ollama is running in the background.

## Usage

1. Start the backend:
   ```bash
   python -m uvicorn backend.backend:app --reload --host 0.0.0.0 --port 8000
   ```
2. Start the frontend:
   ```bash
   BACKEND_URL=http://127.0.0.1:8000 python frontend/main.py
   ```

## Docker (Local)

1. Ensure your `.env` is present at the project root (contains RF_API_KEY, SUPABASE_URL, SUPABASE_ANON_KEY, OCR_API_URL, OCR_TOKEN, SUPABASE_SERVICE_ROLE_KEY, etc.).
2. Build and run with Docker Compose:
   ```bash
   docker compose up --build
   ```
   - Backend: http://localhost:8000
   - Frontend (Gradio): http://localhost:7860
   - Ollama is started as a service. Pull the model in a separate terminal once (first run only):
     ```bash
     docker exec -it <compose_project>_ollama_1 ollama pull llama3.2:3b
     ```
   - Alternatively, comment out the `ollama` service in `docker-compose.yml` and set `OLLAMA_API_URL` to your external Ollama host.


## Features

- **Plate Detection**: Uses YOLOv11 via Roboflow to locate plates.
- **Character Recognition**: Uses PaddleOCR for precise alphanumeric extraction.
- **Database Integration**: Looks up driver details in Supabase.
- **AI Assistant**: A built-in chatbot powered by local **Ollama (llama3.2:3b)** that can query your database using natural language.
