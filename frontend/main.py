import gradio as gr
import requests
import os

API_URL = "http://127.0.0.1:8000/detect"

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
            # Added timeout to avoid hanging indefinitely
            response = requests.post(API_URL, files=files, timeout=60)
            
        print(f"Backend responded with status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            if "error" in result:
                print(f"Backend returned error: {result['error']}")
                return result["error"]
            
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

demo = gr.Interface(
    fn=detect_plate,
    inputs=gr.Image(type="pil"),
    outputs=gr.Textbox(label="Detection Result"),
    title="Tunisian License Plate Reader"
)

if __name__ == "__main__":
    demo.launch()
