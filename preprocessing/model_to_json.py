import os
import json
import base64
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

def get_image_size(image_path):
    image = cv2.imread(image_path)
    return image.shape[1], image.shape[0]  # width, height

def mask_to_polygon(mask, epsilon=5.0, min_points=4):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if len(contour) >= 3:  # valid polygon has at least 3 points
            simplified_contour = cv2.approxPolyDP(contour, epsilon, True)
            if len(simplified_contour) >= min_points:  # filter polygons with fewer points
                polygon = []
                for point in simplified_contour:
                    polygon.append([float(point[0][0]), float(point[0][1])])
                polygons.append(polygon)
    return polygons

def format_points(points):
    return [[round(point[0], 6), round(point[1], 6)] for point in points]

def convert_to_labelme_format(image_path, results):
    shapes = []
    if hasattr(results[0], 'masks') and results[0].masks.data is not None:
        for mask_data, class_id in zip(results[0].masks.data, results[0].boxes.cls):
            mask = mask_data.cpu().numpy().astype(np.uint8)
            label = results[0].names[int(class_id)]
            polygons = mask_to_polygon(mask)
            for polygon in polygons:
                formatted_polygon = format_points(polygon)
                shape = {
                    "label": label,
                    "points": formatted_polygon,
                    "group_id": None,
                    "description": "",
                    "shape_type": "polygon",
                    "flags": {},
                    "mask": None
                }
                shapes.append(shape)
    
    width, height = get_image_size(image_path)
    
    data = {
        "version": "5.4.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": os.path.join("..\\images", os.path.basename(image_path)),
        "imageData": None,  # Set to None to be serialized as null in JSON
        "imageHeight": height,
        "imageWidth": width
    }
    
    return data

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Initialize the model
model = YOLO('./model/best_segmentation.pt')
test_image_dir = './images'
annotation_save_dir = './annotation_generator'
os.makedirs(annotation_save_dir, exist_ok=True)

test_images = [os.path.join(test_image_dir, img) for img in os.listdir(test_image_dir) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Predict and save results for each image
for image_path in test_images:
    if not os.path.isfile(image_path):
        print(f"File not found: {image_path}")
        continue

    results = model.predict(source=image_path, imgsz=1024)  # predict on an image

    # Convert to LabelMe JSON format and save
    labelme_data = convert_to_labelme_format(image_path, results)
    json_output_path = os.path.join(annotation_save_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}.json")
    with open(json_output_path, 'w') as json_file:
        json.dump(labelme_data, json_file, indent=4)
    print(f"Saved LabelMe JSON annotation to {json_output_path}")
