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

def mask_to_polygon(mask, epsilon=5.0):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if len(contour) >= 3:  # valid polygon has at least 3 points
            simplified_contour = cv2.approxPolyDP(contour, epsilon, True)
            polygon = []
            for point in simplified_contour:
                polygon.append([float(point[0][0]), float(point[0][1])])
            polygons.append(polygon)
    return polygons

def image_to_base64(image_path):
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def draw_endpoints(image, polygons):
    for polygon in polygons:
        for point in polygon:
            cv2.circle(image, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)
    return image

def convert_to_labelme_format(image_path, results):
    shapes = []
    if hasattr(results[0], 'masks') and results[0].masks.data is not None:
        for mask_data in results[0].masks.data:
            mask = mask_data.cpu().numpy().astype(np.uint8)
            polygons = mask_to_polygon(mask)
            for polygon in polygons:
                shape = {
                    "label": "wall",
                    "points": polygon,
                    "group_id": None,
                    "description": "",
                    "shape_type": "polygon",
                    "flags": {},
                    "mask": None
                }
                shapes.append(shape)
    
    width, height = get_image_size(image_path)
    image_data = image_to_base64(image_path)
    
    data = {
        "version": "5.4.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": os.path.basename(image_path),
        "imageData": image_data,
        "imageHeight": height,
        "imageWidth": width
    }
    
    return data

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Initialize the model
model = YOLO('./runs/segment/train/weights/best.pt')
test_image_dir = './test_img'
annotation_save_dir = './annotation_generator'
save_dir = './output'
os.makedirs(annotation_save_dir, exist_ok=True)
os.makedirs(save_dir, exist_ok=True)

test_images = [os.path.join(test_image_dir, img) for img in os.listdir(test_image_dir) if img.endswith(('.PNG', '.jpg', '.jpeg'))]

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

    # Visualize and save only the endpoints on segmentation
    img_array = cv2.imread(image_path)
    if hasattr(results[0], 'masks') and results[0].masks.data is not None:
        for mask_data in results[0].masks.data:
            mask = mask_data.cpu().numpy().astype(np.uint8)
            polygons = mask_to_polygon(mask)
            img_with_endpoints = draw_endpoints(img_array, polygons)

            # Convert the array to an image and save it
            img = Image.fromarray(cv2.cvtColor(img_with_endpoints, cv2.COLOR_BGR2RGB))
            output_path = os.path.join(save_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_endpoints.png")
            img.save(output_path)
            print(f"Saved mask overlay with endpoints to {output_path}")
