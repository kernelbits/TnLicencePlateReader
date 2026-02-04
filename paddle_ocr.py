import base64
import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
API_URL = os.getenv("OCR_API_URL")
TOKEN = os.getenv("OCR_TOKEN")
file_path = "data/test_image2.jpg"

# Prepare image data
with open(file_path, "rb") as file:
    file_bytes = file.read()
    file_data = base64.b64encode(file_bytes).decode("ascii")

headers = {
    "Authorization": f"token {TOKEN}",
    "Content-Type": "application/json"
}

payload = {
    "file": file_data,
    "fileType": 1,  # For plate/image
    }

response = requests.post(API_URL, json=payload, headers=headers)
print("HTTP status:", response.status_code)
assert response.status_code == 200

result = response.json()["result"]

# --- Begin Plate Extraction ---
layout_results = result.get("layoutParsingResults", [])
if layout_results:
    for layout in layout_results:
        # Try to get block_content from parsing_res_list in prunedResult
        pruned = layout.get("prunedResult", {})
        parsing_res_list = pruned.get("parsing_res_list", [])
        for block in parsing_res_list:
            label = block.get("block_label", "")
            content = block.get("block_content", "")
            if label == "paragraph_title" and content:
                print(f"Detected plate: {content.strip()}")
        # Or, fallback to Markdown text
        markdown = layout.get("markdown", {})
        md_plate = markdown.get("text", "").strip()
        if md_plate:
            print(f"Markdown-detected plate: {md_plate.lstrip('#').strip()}")
else:
    print("No layout results found!")
