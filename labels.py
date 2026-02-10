import os

def update_file(file_path):
    """Update the labels in the file to include 'تونس' between numbers."""
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        return
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    updated_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 3:  # image_path series number
            image_path, series, number = parts
            updated_line = f"{image_path} {series} تونس {number}\n"
            updated_lines.append(updated_line)
        else:
            updated_lines.append(line)  # Keep as is if format doesn't match
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(updated_lines)
    
    print(f"Updated {file_path}")

# Update the files
update_file('fineTunePaddleOCR/train.txt')
update_file('fineTunePaddleOCR/val.txt')

print("All files updated!")


