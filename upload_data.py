from roboflow import Roboflow
from dotenv import load_dotenv
import os 
load_dotenv()

WORKSPACE = "itgateinternship"
PROJECT = "tunisian-license-plate-xe5yl-d8jgs"
VERSION = 2

rf = Roboflow(api_key=os.getenv("RF_API_KEY"))
project = rf.workspace(WORKSPACE).project(PROJECT)

train_folder = "data/images"

for filename in os.listdir(train_folder):
    if filename.endswith(".jpg"):
        image_path = os.path.join(train_folder, filename)
        annotation_path = os.path.join(train_folder, filename.replace(".jpg", ".txt"))
        
        if os.path.exists(annotation_path):
            project.upload(
                image_path=image_path,
                annotation_path=annotation_path,
                annotation_format="yolov5"
            )
            print(f"Uploaded: {filename}")
        else:
            print(f"No annotation for: {filename}")
