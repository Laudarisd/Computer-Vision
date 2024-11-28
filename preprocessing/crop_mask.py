import json
import os
from PIL import Image, ImageDraw
from glob import glob
import numpy as np
from collections import Counter

def create_mask_from_polygon(polygon, image_size):
    mask = Image.new('L', image_size, 0)
    flat_polygon = [item for sublist in polygon for item in sublist]
    ImageDraw.Draw(mask).polygon(flat_polygon, outline=1, fill=1)
    return np.array(mask)

def crop_and_save_mask(json_folder, image_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    json_files = glob(os.path.join(json_folder, "*.json"))
    total_shapes = Counter()
    processed_shapes = Counter()
    
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            print(f"Processing file: {json_file}")
        
        image_path = os.path.join(image_folder, os.path.basename(data['imagePath']))
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue
        
        img = Image.open(image_path)
        img_array = np.array(img)
            
        shape_counter = 0
        
        for shape in data['shapes']:
            class_name = shape['label']
            total_shapes[class_name] += 1
            points = shape['points']
            
            try:
                mask = create_mask_from_polygon(points, img.size)
            except ValueError as e:
                print(f"Error creating mask for {class_name} in {data['imagePath']}: {e}")
                continue
            
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            if not np.any(rows) or not np.any(cols):
                print(f"Empty mask for {class_name} in {data['imagePath']}")
                continue
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]
            
            cropped_img_array = img_array[y_min:y_max+1, x_min:x_max+1]
            mask_cropped = mask[y_min:y_max+1, x_min:x_max+1]
            
            if len(cropped_img_array.shape) == 3:  # Color image
                cropped_img_array = cropped_img_array * mask_cropped[:, :, np.newaxis]
            else:  # Grayscale image
                cropped_img_array = cropped_img_array * mask_cropped
            
            cropped_img = Image.fromarray(cropped_img_array.astype(np.uint8))
            
            new_width = int((x_max - x_min) * 2)
            new_height = int((y_max - y_min) * 2)
            
            cropped_img.thumbnail((new_width, new_height), Image.LANCZOS)
            
            class_folder = os.path.join(output_folder, class_name)
            os.makedirs(class_folder, exist_ok=True)
            
            shape_counter += 1
            output_filename = f"{os.path.splitext(os.path.basename(data['imagePath']))[0]}_{class_name}_{shape_counter}.png"
            output_path = os.path.join(class_folder, output_filename)
            
            try:
                cropped_img.save(output_path)
                processed_shapes[class_name] += 1
                print(f"Saved: {output_path}")
            except Exception as e:
                print(f"Error saving {output_path}: {e}")

    print("\nProcessing Summary:")
    print("Class\t\tTotal\tProcessed")
    for class_name in total_shapes:
        print(f"{class_name}\t\t{total_shapes[class_name]}\t{processed_shapes[class_name]}")

json_dir = './json'
img_dir = './images'
output_dir = './cropped'
crop_and_save_mask(json_dir, img_dir, output_dir)
