import os
import easyocr
import random

# Folder with cropped plates
folder = "cropped_plates"
output_train_file = "rec_gt_train.txt"
output_val_file = "rec_gt_val.txt"

# Initialize EasyOCR only for digits (English only, as you specified)
reader = easyocr.Reader(['en'], gpu=False)

images = sorted([f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png"))])
total = len(images)
print(f"Found {total} images. Generating ground truth file...")

# Collect data
data = []
for i, img_file in enumerate(images, 1):
    img_path = os.path.join(folder, img_file)
    result = reader.readtext(img_path)
    
    detected_text = ""
    for (_, text, _) in result:
        # Keep only digits
        digits = "".join(c for c in text if c.isdigit())
        detected_text += digits

    # Force the format: first 3 digits = series, last 4 digits = number
    series = detected_text[:3].zfill(3)  # pad to 3 digits
    number = detected_text[-4:].zfill(4)  # pad to 4 digits
    label_text = f"{series} {number}"

    # Add to data (you can add Arabic here if preferred, e.g., f"{series} تونس {number}")
    data.append(f"{img_path}\t{label_text}")
    
    if i % 50 == 0 or i == total:
        print(f"[{i}/{total}] {img_file} -> {label_text}")

# Shuffle and split (80% train, 20% val)
random.shuffle(data)
split_idx = int(0.8 * len(data))
train_data = data[:split_idx]
val_data = data[split_idx:]

# Write to files
with open(output_train_file, "w", encoding="utf-8") as f:
    for line in train_data:
        f.write(line + "\n")

with open(output_val_file, "w", encoding="utf-8") as f:
    for line in val_data:
        f.write(line + "\n")

print(f"Training data: {len(train_data)} samples -> {output_train_file}")
print(f"Validation data: {len(val_data)} samples -> {output_val_file}")
print("Ground truth files generated! Open them in a text editor to manually fix any label mistakes, then add Arabic if needed.")