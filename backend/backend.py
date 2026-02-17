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
from pydantic import BaseModel
from typing import Optional, List
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

OLLAMA_API = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")
MODEL_NAME = "llama3.2:3b"

class ChatRequest(BaseModel):
    message: str
    context_plate: Optional[str] = None
    history: Optional[List[dict]] = None

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    print(f"\n--- New Chat Request ---")
    print(f"User Message: {request.message}")
    
    # 1. Ask Ollama to interpret the message and decide if it needs a database query
    schema_context = """
    Database Tables:
    1. license_plates:
       - plate_number (text, primary key, e.g., '125تونس8365')
       - series (text, e.g., '125')
       - number (text, e.g., '8365')
       - driver_name (text)
       - driver_id (text)
       - address (text)
       - vehicle_make (text, e.g., 'Ford', 'Toyota')
       - vehicle_model (text)
       - vehicle_year (integer)
       - registration_date (date)
       - expiry_date (date)
       - status (text, e.g., 'valid', 'expired')
       - violations (integer)
       - notes (text)
    2. detection_logs:
       - id (bigint, primary key)
       - plate_number (text, references license_plates)
       - image_url (text)
       - created_at (timestamp)
    """
    
    prompt = f"""
    You are an AI assistant for the Tunisian License Plate Reader system.
    {schema_context}
    
    Current plate in focus (if any): {request.context_plate or "None"}
    
    User question: {request.message}
    
    Instructions:
    - If you need data from the database, you MUST generate a JSON query specification.
    - DO NOT explain your reasoning or SQL logic to the user.
    - DO NOT use joins. We only support single-table queries on 'license_plates' or 'detection_logs'.
    - If the user asks for "all" or multiple records, use the 'limit' parameter (default 10, max 100).
    - For partial matches (e.g., car makes, names), use the 'ilike' operator.
    - IMPORTANT: When searching for names, use 'ilike' with 'driver_name'.
    - IMPORTANT: If the user says "don't focus on current plate" or asks a general question about other people, ignore the 'Current plate in focus'.
    
    Response format:
    ACTION: QUERY
    DATA: {{"table": "...", "select": "*", "filters": [{{"col": "...", "op": "...", "val": "..."}}], "limit": 10}}
    
    OR if you can answer directly without the database:
    ACTION: ANSWER
    DATA: Your natural language response.
    
    CRITICAL: 
    - NEVER return raw JSON to the user as a final answer.
    - If you use ACTION: QUERY, only output the JSON block after DATA:. No conversational filler.
    - Example for "Who drives Ford cars?":
      ACTION: QUERY
      DATA: {{"table": "license_plates", "select": "driver_name", "filters": [{{"col": "vehicle_make", "op": "ilike", "val": "Ford"}}]}}
    - Example for "Is there a driver named Hamed?":
      ACTION: QUERY
      DATA: {{"table": "license_plates", "select": "*", "filters": [{{"col": "driver_name", "op": "ilike", "val": "Hamed"}}]}}
    """
    
    try:
        # Check if Ollama is running
        try:
            ollama_res = requests.post(OLLAMA_API, json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False
            }, timeout=10)
        except requests.exceptions.ConnectionError:
            return JSONResponse(status_code=503, content={"error": "Ollama service is not running. Please start Ollama."})

        if ollama_res.status_code != 200:
             return JSONResponse(status_code=500, content={"error": f"Ollama error: {ollama_res.text}"})
        
        ai_response = ollama_res.json().get("response", "")
        print(f"AI Response Raw: {ai_response}")
        
        # Parse AI response
        if "ACTION: QUERY" in ai_response or (ai_response.strip().startswith("{") and "table" in ai_response):
            import re
            # Try to find JSON in the response
            json_match = re.search(r"(\{.*\})", ai_response, re.DOTALL)
            if json_match:
                try:
                    query_info = json.loads(json_match.group(1))
                    table = query_info.get("table", "license_plates")
                    select = query_info.get("select", "*")
                    filters = query_info.get("filters", [])
                    
                    print(f"AI decided to query {table} (select {select}) with filters {filters}")
                    
                    if supabase:
                        # Construct query
                        query = supabase.table(table).select(select)
                        for f in filters:
                            col = f.get("col")
                            op = f.get("op", "eq")
                            val = f.get("val")
                            
                            if not col or val is None: continue
                            
                            if op == "eq": query = query.eq(col, val)
                            elif op == "neq": query = query.neq(col, val)
                            elif op == "gt": query = query.gt(col, val)
                            elif op == "lt": query = query.lt(col, val)
                            elif op == "gte": query = query.gte(col, val)
                            elif op == "lte": query = query.lte(col, val)
                            elif op == "like": query = query.like(col, val)
                            elif op == "ilike": query = query.ilike(col, f"%{val}%")
                        
                        limit = query_info.get("limit")
                        if limit:
                            query = query.limit(int(limit))
                        
                        print(f"Executing query on Supabase...")
                        db_res = query.execute()
                        results = db_res.data
                        print(f"Database results found: {len(results)}")
                        
                        # Final processing by AI
                        final_prompt = f"""
                        You are an AI assistant for the Tunisian License Plate Reader system.
                        User asked: {request.message}
                        Database results: {json.dumps(results)}
                        
                        Instructions:
                        1. Provide a friendly natural language answer based ONLY on the database results above.
                        2. If the results are empty, say that no matching records were found.
                        3. For lists, use bullet points.
                        4. Do NOT mention "JSON", "query", "database", or "ACTION". 
                        5. Do NOT show the raw data. Just the answer.
                        """
                        
                        final_res = requests.post(OLLAMA_API, json={
                            "model": MODEL_NAME,
                            "prompt": final_prompt,
                            "stream": False
                        }, timeout=30)
                        
                        answer = final_res.json().get("response", "Error processing final answer.")
                        return {"answer": answer, "data": results}
                    else:
                        return {"answer": "Database connection not available.", "data": []}
                except Exception as parse_err:
                    print(f"Error parsing AI query JSON: {parse_err}")
                    # Fall back to returning the raw response
        
        # If not a query or if parsing failed, return the AI response directly
        answer = ai_response.replace("ACTION: ANSWER", "").replace("DATA:", "").strip()
        return {"answer": answer}

    except Exception as e:
        print(f"Chat error: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

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
        OCR_TIMEOUT = 60
        
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