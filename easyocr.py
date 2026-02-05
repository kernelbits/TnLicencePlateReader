import os
import cv2
import easyocr

CROPPED_FOLDER = "cropped_plates"      # folder with cropped plate images
PROCESSED_FOLDER = "processed_plates"  # temp folder for thresholded images (optional)
LABELS_FILE = "labels.txt"             # output file
APPLY_THRESHOLD = True                  # apply threshold to make numbers clearer

os.makedirs(PROCESSED_FOLDER, exist_ok=True)

reader = easyocr.Reader(['en'], gpu=False)  # English digits only

labels = {}
image_files = [f for f in os.listdir(CROPPED_FOLDER) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

for idx, f in enumerate(image_files):
    img_path = os.path.join(CROPPED_FOLDER, f)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  
    
    if APPLY_THRESHOLD:
        _, img_thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    else:
        img_thresh = img

    processed_path = os.path.join(PROCESSED_FOLDER, f)
    cv2.imwrite(processed_path, img_thresh)

    result = reader.readtext(img_thresh, allowlist='0123456789')
    numbers = ''.join([r[1] for r in result])  # concatenate all detected numbers

    labels[f] = numbers
    print(f"[{idx + 1}/{len(image_files)}] {f} -> {numbers}")

with open(LABELS_FILE, "w", encoding="utf-8") as f:
    for img_name, numbers in labels.items():
        f.write(f"{os.path.join(PROCESSED_FOLDER, img_name)}\t{numbers}\n")

print(f"\nDone! Labels saved to '{LABELS_FILE}'")
