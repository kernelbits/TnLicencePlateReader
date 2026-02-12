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

## Usage

1. Start the backend:
   ```bash
   python backend/backend.py
   ```
2. Start the frontend:
   ```bash
   python frontend/main.py
   ```
