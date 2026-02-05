import os
import shutil
from pathlib import Path
from PIL import Image
from roboflow import Roboflow
import time

DATA_DIR = Path("data")           # Folder with images to process
PLATES_DIR = Path("cropped_plates")       # Folder to save cropped plates
FAILED_DIR = Path("failed")       # Folder for images with no plates
CONFIDENCE_THRESHOLD = 0.6        # Minimum confidence to consider detection
OVERLAP = 0.3                     # Overlap for Roboflow prediction (if used)
RETRY_DELAY = 5                    # Seconds to wait on network/API error

PLATES_DIR.mkdir(exist_ok=True)
FAILED_DIR.mkdir(exist_ok=True)

WORKSPACE = "itgateinternship"
PROJECT = "tunisian-license-plate-xe5yl-d8jgs"
VERSION = 2

api_key = os.environ.get("RF_API_KEY")

workspace = os.environ.get("WORKSPACE", WORKSPACE)
project_name = os.environ.get("PROJECT", PROJECT)
model_version = int(os.environ.get("MODEL_VERSION", VERSION))

if not api_key or not workspace or not project_name:
    raise ValueError("Missing RF_API_KEY, WORKSPACE, or PROJECT env variables!")

rf = Roboflow(api_key=api_key)
project = rf.workspace(workspace).project(project_name)
model = project.version(model_version).model

image_files = sorted(DATA_DIR.glob("*.*"))
total_images = len(image_files)

for idx, image_path in enumerate(image_files, 1):
    print(f"[{idx}/{total_images}] Processing: {image_path.name}")

    success = False
    retries = 0
    while not success:
        try:
            prediction = model.predict(str(image_path), confidence=CONFIDENCE_THRESHOLD, overlap=OVERLAP)
            success = True
        except Exception as e:
            retries += 1
            print(f"  Roboflow API error: {e} | retrying in {RETRY_DELAY}s... (Attempt {retries})")
            time.sleep(RETRY_DELAY)

    predictions = prediction.json().get("predictions", [])

    if not predictions:
        print("  No plates detected, moving image to failed folder.")
        shutil.move(str(image_path), FAILED_DIR / image_path.name)
        continue

    img = Image.open(image_path)

    for i, p in enumerate(predictions, 1):
        x, y, w, h = p["x"], p["y"], p["width"], p["height"]

        left = int(x - w / 2)
        top = int(y - h / 2)
        right = int(x + w / 2)
        bottom = int(y + h / 2)

        cropped = img.crop((left, top, right, bottom))
        cropped_filename = PLATES_DIR / f"{image_path.stem}_plate{i}{image_path.suffix}"
        cropped.save(cropped_filename)

    image_path.unlink()  
print("Processing complete!")
