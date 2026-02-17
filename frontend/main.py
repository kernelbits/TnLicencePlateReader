import gradio as gr
import requests
import os

BACKEND_BASE_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
API_URL = f"{BACKEND_BASE_URL}/detect"
CHAT_API_URL = f"{BACKEND_BASE_URL}/chat"

# To keep track of the last detected plate for chat context
state = {"last_plate": None}

def detect_plate(image):
    if image is None:
        return "No image provided"
    
    print(f"--- Frontend: New Detection Request ---")
    # Save image to temp file for sending
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
                return result["error"]
            
            plate = result.get("plate_number", "Unknown")
            state["last_plate"] = plate # Store for chat context
            
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
            
            print(f"Successfully processed result for plate: {plate}")
            return info
        else:
            error_msg = f"Error: {response.status_code} - {response.text}"
            print(f"Backend error: {error_msg}")
            return error_msg
    except requests.exceptions.Timeout:
        print("Frontend Error: Request to backend timed out")
        return "Error: Request to backend timed out. Please try again."
    except Exception as e:
        print(f"Frontend Error: {str(e)}")
        return f"Error connecting to backend: {str(e)}"
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                print(f"Cleaned up frontend temp file: {temp_path}")
            except:
                pass

def chat_response(message, history):
    payload = {
        "message": message,
        "context_plate": state["last_plate"]
    }
    try:
        response = requests.post(CHAT_API_URL, json=payload, timeout=60)
        if response.status_code == 200:
            return response.json().get("answer", "No response from AI.")
        elif response.status_code == 503:
            return "Ollama is not running. Please start it to use the chatbot."
        else:
            return f"Error from backend: {response.text}"
    except Exception as e:
        return f"Error connecting to backend: {str(e)}"

with gr.Blocks() as demo:
    gr.Markdown("# Tunisian License Plate Reader & Assistant")
    
    with gr.Row():
        with gr.Column(scale=2):
            with gr.Group():
                gr.Markdown("### 📸 Plate Detection")
                input_img = gr.Image(type="pil")
                btn = gr.Button("Detect Plate", variant="primary")
                output_text = gr.Textbox(label="Detection Result", lines=10)
        
        with gr.Column(scale=3):
            with gr.Group():
                gr.Markdown("### 🤖 AI Database Assistant")
                # Using ChatInterface directly in Blocks
                gr.ChatInterface(
                    fn=chat_response
                )
    
    btn.click(fn=detect_plate, inputs=input_img, outputs=output_text)

if __name__ == "__main__":
    demo.launch()
