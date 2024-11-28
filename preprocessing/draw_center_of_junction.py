import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET

# Directories for images and XML annotations
img_dir = './images'
xml_dir = './xmls'

# Classes of interest
classes_of_interest = ['junc_I_normal', 'junc_I_open', 'junc_I_isolation', 'junc_L', 'junc_T', 'junc_X']

def draw_circle_outline(img, center, radius, color, thickness=1):
    rows, cols = img.shape[:2]
    y, x = np.ogrid[:rows, :cols]
    dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    
    mask = np.abs(dist_from_center - radius) <= thickness / 2
    img[mask] = color

def draw_bounding_boxes(img_path, boxes):
    img = cv2.imread(img_path)
    for box in boxes:
        # Extract bounding box coordinates and center
        x_min, y_min, x_max, y_max, center_x, center_y, label = box
        # Draw only the circle boundary
        draw_circle_outline(img, (center_x, center_y), 3, (0, 0, 255), 1)
    # Save the image with bounding boxes
    cv2.imwrite(img_path, img)

def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes = []
    for obj in root.findall('object'):
        label = obj.find('name').text
        if label in classes_of_interest:
            xml_box = obj.find('bndbox')
            x_min = int(xml_box.find('xmin').text)
            y_min = int(xml_box.find('ymin').text)
            x_max = int(xml_box.find('xmax').text)
            y_max = int(xml_box.find('ymax').text)
            center_x = (x_min + x_max) // 2
            center_y = (y_min + y_max) // 2
            boxes.append((x_min, y_min, x_max, y_max, center_x, center_y, label))
    return boxes

def process_images_and_annotations(img_dir, xml_dir):
    img_list = os.listdir(img_dir)
    for img_name in img_list:
        img_path = os.path.join(img_dir, img_name)
        xml_name = os.path.splitext(img_name)[0] + '.xml'
        xml_path = os.path.join(xml_dir, xml_name)
        
        if os.path.exists(xml_path):
            boxes = parse_xml(xml_path)
            draw_bounding_boxes(img_path, boxes)

if __name__ == '__main__':
    process_images_and_annotations(img_dir, xml_dir)