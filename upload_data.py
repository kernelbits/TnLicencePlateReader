from roboflow import Roboflow
from dotenv import load_dotenv
import os 
load_dotenv()
#Loading our lib and env variables 


rf = Roboflow(api_key=os.getenv("RF_API_KEY"))
project = rf.workspace("itgateinternship").project("tunisian-license-plate-xe5yl-d8jgs")
#Accessing our project

train_folder = "data/train"

# Get all jpg files
for filename in os.listdir(train_folder):
    if filename.endswith(".jpg"):
        image_path = os.path.join(train_folder, filename)
        annotation_path = os.path.join(train_folder, filename.replace(".jpg", ".txt"))
        
        # Check if annotation exists
        if os.path.exists(annotation_path):
            project.upload(
                image_path=image_path,
                annotation_path=annotation_path,
                annotation_format="yolov5"
            )
            print(f"Uploaded: {filename}")
        else:
            print(f"No annotation for: {filename}")


"""
PreProcessing Applied : 
    auto_orient:
    resize: stretch to 640*640

Augmentations Applied :
    flip: horizontal
    brightness: between -25% and +25%
    blur: 1.5px



Training Set: 1.5k images
Validation Set: 347 images
Testing Set: 276 images



YOLOv11 - fast 
"""
