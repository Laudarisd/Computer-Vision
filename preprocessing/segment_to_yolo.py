import json
import os
from glob import glob

def normalize_coordinates(points, img_width, img_height):
    return [(x / img_width, y / img_height) for x, y in points]

def converter_labelme_to_yolo(json_path, output_dir, class_map):
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    img_width = data['imageWidth']
    img_height = data['imageHeight']
    
    filename = os.path.splitext(os.path.basename(data['imagePath']))[0]
    output_path = os.path.join(output_dir, f"{filename}.txt")
    
    with open(output_path, 'w') as f:
        for shape in data['shapes']:
            label = shape['label']
            if label not in class_map:
                print(f"Warning: Label '{label}' not in class map. Skipping.")
                continue
            class_id = class_map[label]
            points = shape['points']
            
            normalized_points = normalize_coordinates(points, img_width, img_height)
            flattened_points = [coord for point in normalized_points for coord in point]
            line = f"{class_id} " + " ".join(map(str, flattened_points))
            f.write(line + '\n')

def main():
    json_dir = "./json"
    output_dir = './yolo_format'
    
    # Define class map
    class_map = {
        "wall": 0,
        "bed_room": 1, 
        "bathroom": 2, 
        "balcony": 3,
        "entrance": 4,
        "elevator": 5, 
        "dressing_room": 6,
        "air_room": 7,
        "utility_room": 8, 
        "pantry": 9, 
        "hallway": 10, 
        "stairs": 11,
        "living_kitchen": 12,
        "others": 13
    }
    
    os.makedirs(output_dir, exist_ok=True)
    
    json_files = glob(os.path.join(json_dir, '*.json'))
    for json_file in json_files:
        try:
            converter_labelme_to_yolo(json_file, output_dir, class_map)
            print(f"Converted {json_file}")
        except Exception as e:
            print(f"Error processing {json_file}: {str(e)}")
        
if __name__ == "__main__":
    main()
