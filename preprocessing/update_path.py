import os
import json

json_dir = './original_annotation'

def update_image_path(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if "imagePath" in data:
        original_path = data["imagePath"]
        file_name = os.path.basename(original_path)
        new_path = f"../images/{file_name}"
        data["imagePath"] = new_path

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)


for file_name in os.listdir(json_dir):
    if file_name.endswith('.json'):
        file_path = os.path.join(json_dir, file_name)
        update_image_path(file_path)

print("Image paths updated successfully!")