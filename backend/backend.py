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
import json
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

app = FastAPI()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    print("WARNING: SUPABASE_URL or SUPABASE_ANON_KEY is not set in .env file")

supabase = None
if SUPABASE_URL:
    key = SUPABASE_SERVICE_ROLE_KEY or SUPABASE_ANON_KEY
    if key:
        supabase = create_client(SUPABASE_URL, key)


# PaddleOCR API
API_URL = os.getenv("OCR_API_URL")
TOKEN = os.getenv("OCR_TOKEN")

if not API_URL or not TOKEN:
    print("WARNING: OCR_API_URL or OCR_TOKEN is not set in .env file")

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
    print(f"\n--- New Detection Request: {image.filename} ---")
    image_data = await image.read()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(image_data)
        temp_path = temp_file.name
    
    try:
        print(f"Opening image: {temp_path}")
        img = Image.open(temp_path)
        img = img.resize((640, 640))
        img.save(temp_path)
        
        # Predict
        print("Running Roboflow prediction...")
        success = False
        retries = 0
        while not success and retries < 3:
            try:
                prediction = model.predict(temp_path, confidence=CONFIDENCE_THRESHOLD, overlap=OVERLAP)
                success = True
                print("Roboflow prediction successful")
            except Exception as e:
                retries += 1
                print(f"Roboflow retry {retries}/3 after error: {e}")
                time.sleep(RETRY_DELAY)
        
        if not success:
            return JSONResponse(status_code=500, content={"error": "Prediction failed after retries"})
        
        predictions = prediction.json().get("predictions", [])
        
        if not predictions:
            print("No plates detected in image")
            return JSONResponse(status_code=404, content={"error": "No plates detected"})
        
        print(f"Detected {len(predictions)} potential plate(s)")
        p = predictions[0]
        x, y, w, h = p["x"], p["y"], p["width"], p["height"]
        left = int(x - w / 2)
        top = int(y - h / 2)
        right = int(x + w / 2)
        bottom = int(y + h / 2)
        
        cropped = img.crop((left, top, right, bottom))
        
        # PaddleOCR
        print("Preparing for PaddleOCR...")
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
        
        print(f"Calling PaddleOCR API: {API_URL}")
        
        success_ocr = False
        retries_ocr = 0
        OCR_TIMEOUT = 60 # Increased timeout
        
        while not success_ocr and retries_ocr < 3:
            try:
                ocr_response = requests.post(API_URL, json=payload, headers=headers, timeout=OCR_TIMEOUT)
                if ocr_response.status_code == 200:
                    success_ocr = True
                else:
                    retries_ocr += 1
                    print(f"PaddleOCR API failed with status {ocr_response.status_code}. Retry {retries_ocr}/3...")
                    time.sleep(RETRY_DELAY)
            except requests.exceptions.Timeout:
                retries_ocr += 1
                print(f"PaddleOCR API timed out. Retry {retries_ocr}/3...")
                time.sleep(RETRY_DELAY)
            except Exception as e:
                retries_ocr += 1
                print(f"PaddleOCR error: {e}. Retry {retries_ocr}/3...")
                time.sleep(RETRY_DELAY)

        if not success_ocr:
            print("PaddleOCR failed after multiple attempts")
            return JSONResponse(status_code=502, content={"error": "OCR processing failed after retries"})
        
        ocr_json = ocr_response.json()
        result = ocr_json.get("result", {})
        if not result.get("layoutParsingResults"):
            print("No layout parsing results from OCR")
            return JSONResponse(status_code=422, content={"error": "No OCR text found"})
        
        raw_text = result["layoutParsingResults"][0].get("markdown", {}).get("text", "")
        if not raw_text:
             print("Raw OCR text is empty")
             return JSONResponse(status_code=422, content={"error": "No text found in OCR result"})
        
        print(f"OCR Raw Text: {raw_text}")
        digits = "".join(c for c in raw_text if c.isdigit())
        # Formulate plate number with Arabic characters
        plate_number = f"{digits[:3].zfill(3)}تونس{digits[-4:].zfill(4)}"
        print(f"Processed Plate Number: {plate_number}")
        
        print("Querying database for driver info...")
        db_result = query_database(plate_number)
        
        # Log detection to Supabase
        print("Logging detection and uploading cropped image...")
        try:
            image_url = log_detection(plate_number, cropped_data)
        except Exception as e:
            print(f"Logging failed: {e}")
            image_url = None

        # Ensure image_url is JSON serializable (string or None)
        if image_url is not None:
            image_url = str(image_url)

        response_data = {
            "plate_number": plate_number,
            "driver_info": db_result[0] if (db_result and len(db_result) > 0) else None,
            "image_url": image_url
        }
        
        print(f"Success! Returning response: {response_data}")
        # Explicitly use jsonable_encoder to prevent serialization errors
        return JSONResponse(content=jsonable_encoder(response_data))
        
    except Exception as e:
        print(f"CRITICAL ERROR in /detect: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        if 'temp_path' in locals() and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
                print(f"Cleaned up temp file: {temp_path}")
            except:
                pass

def query_database(plate_number):
    if not supabase:
        return []
    try:
        response = supabase.table("license_plates").select("*").eq("plate_number", plate_number).execute()
        return response.data
    except Exception as e:
        print(f"Database query error: {e}")
        return []

def log_detection(plate_number, image_bytes):
    if not supabase:
        return None
    
    try:
        # 1. Upload cropped image to 'plates' bucket
        # Sanitize plate_number for filename (remove Arabic characters or non-ASCII)
        safe_plate = "".join(c for c in plate_number if c.isalnum() and ord(c) < 128)
        file_name = f"detection_{int(time.time())}_{safe_plate}.jpg"
        bucket_name = "plates"
        
        # Supabase storage upload
        supabase.storage.from_(bucket_name).upload(
            path=file_name,
            file=image_bytes,
            file_options={"content-type": "image/jpeg"}
        )
        
        # 2. Get Public URL
        # We ensure the bucket is public so this URL is accessible
        res = supabase.storage.from_(bucket_name).get_public_url(file_name)
        
        # Handle different return types from get_public_url depending on version
        if isinstance(res, str):
            image_url = res
        elif hasattr(res, "public_url"):
            image_url = res.public_url
        elif isinstance(res, dict):
            image_url = res.get("publicURL") or res.get("public_url") or str(res)
        else:
            image_url = str(res)

        # 3. Insert into detection_logs
        log_data = {
            "plate_number": plate_number,
            "image_url": image_url,
        }
        
        # Try inserting - if the table doesn't exist, this will fail but we still return the image_url
        try:
            # Convert any non-serializable objects in log_data
            serializable_log_data = {
                "plate_number": str(plate_number),
                "image_url": str(image_url) if image_url else None
            }
            print(f"Attempting to insert log: {serializable_log_data}")
            supabase.table("detection_logs").insert(serializable_log_data).execute()
            print("Insert successful")
        except Exception as e:
            print(f"Error inserting into detection_logs: {e}")
            # The table might not exist yet, but the image is uploaded
            
        return image_url
        
    except Exception as e:
        print(f"Error in log_detection: {e}")
        return None