import gradio as gr
import requests
import os
from PIL import ImageDraw

BACKEND_BASE_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
API_URL = f"{BACKEND_BASE_URL}/detect"

def detect_plate(image):
    if image is None:
        return "No image provided", None
    
    print(f"--- Frontend: New Detection Request ---")
    temp_path = "temp_upload.jpg"
    image.save(temp_path)
    
    try:
        print(f"Sending image to backend: {API_URL}")
        with open(temp_path, "rb") as f:
            files = {"image": f}
            response = requests.post(API_URL, files=files, timeout=60)
            
        print(f"Backend responded with status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            if "error" in result:
                print(f"Backend returned error: {result['error']}")
                return result["error"], image
            
            plate = result.get("plate_number", "Unknown")
            driver = result.get("driver_info")
            
            info = f"Plate Number: {plate}\n"
            if driver:
                info += f"Driver: {driver.get('driver_name', 'N/A')}\n"
                info += f"Car: {driver.get('vehicle_make', 'N/A')}\n"
            else:
                info += "Driver info not found in database.\n"
            
            image_url = result.get("image_url")
            if image_url:
                info += f"\nCropped Image: {image_url}"
            
            vision = result.get("vision_validation")
            if vision:
                if "error" in vision:
                    info += f"\n\nAI Validation Error: {vision['error']}"
                else:
                    info += f"\n\nAI Validation: {vision.get('message', 'No message')}"
                    if not vision.get("match"):
                         info += f"\n   (AI Raw: {vision.get('ai_raw', 'N/A')})"
            
            annotated_image = image.copy()
            predictions = result.get("predictions", [])
            if predictions:
                draw = ImageDraw.Draw(annotated_image)
                orig_w, orig_h = image.size
                scale_x = orig_w / 640.0
                scale_y = orig_h / 640.0
                
                for p in predictions:
                    x, y, w, h = p["x"], p["y"], p["width"], p["height"]
                    left = (x - w / 2) * scale_x
                    top = (y - h / 2) * scale_y
                    right = (x + w / 2) * scale_x
                    bottom = (y + h / 2) * scale_y
                    
                    draw.rectangle([left, top, right, bottom], outline="red", width=5)
            
            print(f"Successfully processed result for plate: {plate}")
            return info, annotated_image
        else:
            error_msg = f"Error: {response.status_code} - {response.text}"
            print(f"Backend error: {error_msg}")
            return error_msg, image
    except requests.exceptions.Timeout:
        print("Frontend Error: Request to backend timed out")
        return "Error: Request to backend timeout. Please try again.", image
    except Exception as e:
        print(f"Frontend Error: {str(e)}")
        return f"Error connecting to backend: {str(e)}", image
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                print(f"Cleaned up frontend temp file: {temp_path}")
            except:
                pass

with gr.Blocks(title="TN License Plate Reader") as demo:
    gr.Markdown("# Tunisian License Plate Reader")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Plate Detection")
            plate_img = gr.Image(type="pil", label="Upload Plate Image Here", interactive=True)
            btn = gr.Button("Detect Plate & Annotate", variant="primary")
        
        with gr.Column(scale=1):
            gr.Markdown("### Results")
            output_text = gr.Textbox(label="Detection Info", lines=18)
    
    btn.click(fn=detect_plate, inputs=plate_img, outputs=[output_text, plate_img])

if __name__ == "__main__":
    demo.launch()
