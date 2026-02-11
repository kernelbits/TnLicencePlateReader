import os
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from supabase import create_client, Client
from pathlib import Path
from PIL import Image
import io
import requests
from roboflow import Roboflow
import time
import tempfile
import base64

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

app = FastAPI()

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_ANON_KEY")
)


# PaddleOCR API
API_URL = "https://x715oe9el5j1ea8a.aistudio-app.com/layout-parsing"
TOKEN = "0533af7020787d282a7baa47d56d21394e18258e"

# Roboflow
WORKSPACE = "itgateinternship"
PROJECT = "tunisian-license-plate-xe5yl-d8jgs"
VERSION = 2
api_key = os.getenv("RF_API_KEY")

rf = Roboflow(api_key=api_key)
project = rf.workspace(WORKSPACE).project(PROJECT)
model = project.version(VERSION).model

CONFIDENCE_THRESHOLD = 0.3
OVERLAP = 0.3
RETRY_DELAY = 5

@app.post("/detect")
async def detect_plates(image: UploadFile = File(...)):
    image_data = await image.read()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(image_data)
        temp_path = temp_file.name
    
    try:
        # Resize
        img = Image.open(temp_path)
        img = img.resize((640, 640))
        img.save(temp_path)
        
        # Predict
        success = False
        retries = 0
        while not success and retries < 3:
            try:
                prediction = model.predict(temp_path, confidence=CONFIDENCE_THRESHOLD, overlap=OVERLAP)
                success = True
            except Exception as e:
                retries += 1
                time.sleep(RETRY_DELAY)
        
        if not success:
            return {"error": "Prediction failed"}
        
        predictions = prediction.json().get("predictions", [])
        
        if not predictions:
            return {"error": "No plates detected"}
        
        p = predictions[0]
        x, y, w, h = p["x"], p["y"], p["width"], p["height"]
        left = int(x - w / 2)
        top = int(y - h / 2)
        right = int(x + w / 2)
        bottom = int(y + h / 2)
        
        cropped = img.crop((left, top, right, bottom))
        
        # PaddleOCR
        cropped_buffer = io.BytesIO()
        cropped.save(cropped_buffer, format="JPEG")
        cropped_data = cropped_buffer.getvalue()
        file_data = base64.b64encode(cropped_data).decode("ascii")
        
        headers = {
            "Authorization": f"token {TOKEN}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "file": file_data,
            "fileType": 1,
            "useDocOrientationClassify": False,
            "useDocUnwarping": False,
            "useChartRecognition": False,
        }
        
        response = requests.post(API_URL, json=payload, headers=headers)
        if response.status_code != 200:
            return {"error": f"PaddleOCR error: {response.status_code}"}
        
        result = response.json()["result"]
        if not result["layoutParsingResults"]:
            return {"error": "No OCR text"}
        
        raw_text = result["layoutParsingResults"][0]["markdown"]["text"]
        digits = "".join(c for c in raw_text if c.isdigit())
        plate_number = f"{digits[:3].zfill(3)}تونس{digits[-4:].zfill(4)}"
        
        db_result = query_database(plate_number)
        

        return {
            "plate_number": plate_number,
            "driver_info": db_result[0] if db_result else None,
        }
    except Exception as e:
        return {"error": str(e)}
    finally:
        os.unlink(temp_path)

def query_database(plate_number):
    response = supabase.table("license_plates").select("*").eq("plate_number", plate_number).execute()
    return response.data