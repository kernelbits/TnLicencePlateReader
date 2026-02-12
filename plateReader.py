from roboflow import Roboflow
from dotenv import load_dotenv
import os

load_dotenv()

WORKSPACE = "itgateinternship"
PROJECT = "tunisian-license-plate-xe5yl-d8jgs"
VERSION = 2


TEST_IMAGE_PATH = "data/test_images/test.jpg"
os.makedirs("predictions", exist_ok=True)


if os.path.exists(TEST_IMAGE_PATH):
    print(f"Test image found at {TEST_IMAGE_PATH}")


rf = Roboflow(api_key=os.getenv("RF_API_KEY"))
project = rf.workspace(WORKSPACE).project(PROJECT)
model = project.version(VERSION).model

prediction = model.predict(TEST_IMAGE_PATH, confidence=40, overlap=30)
prediction.save("predictions/sample1_pred.jpg")
result = prediction.json()


print(result)
