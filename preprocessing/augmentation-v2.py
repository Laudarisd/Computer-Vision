import os
import glob
import cv2
import albumentations as A
import numpy as np
from xml.etree import ElementTree as ET

def load_image_and_xml(image_path, xml_path):
    image = cv2.imread(image_path)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    return image, root

def update_xml_metadata(root, new_image_path):
    # Update filename element
    filename_elem = root.find('filename')
    if filename_elem is not None:
        filename_elem.text = os.path.basename(new_image_path)
    else:
        filename_elem = ET.SubElement(root, 'filename')
        filename_elem.text = os.path.basename(new_image_path)

    # Update path element
    path_elem = root.find('path')
    if path_elem is not None:
        path_elem.text = os.path.abspath(new_image_path)
    else:
        path_elem = ET.SubElement(root, 'path')
        path_elem.text = os.path.abspath(new_image_path)

def save_augmented_image_and_xml(image, root, output_image_path, output_xml_path):
    # Update XML metadata before saving
    update_xml_metadata(root, output_image_path)
    
    # Save image and XML
    cv2.imwrite(output_image_path, image)
    tree = ET.ElementTree(root)
    tree.write(output_xml_path)

def create_augmentation_pipeline(config):
    transforms = []
    if config['augmentations']['RandomBrightnessContrast']['enabled']:
        transforms.append(A.RandomBrightnessContrast(
            brightness_limit=config['augmentations']['RandomBrightnessContrast']['brightness_limit'],
            contrast_limit=config['augmentations']['RandomBrightnessContrast']['contrast_limit'],
            p=config['augmentations']['RandomBrightnessContrast']['p']
        ))
    if config['augmentations']['Sharpen']['enabled']:
        transforms.append(A.Sharpen(p=config['augmentations']['Sharpen']['p']))
    if config['augmentations']['MotionBlur']['enabled']:
        transforms.append(A.MotionBlur(p=config['augmentations']['MotionBlur']['p']))
    if config['augmentations']['HorizontalFlip']['enabled']:
        transforms.append(A.HorizontalFlip(p=config['augmentations']['HorizontalFlip']['p']))
    if config['augmentations']['VerticalFlip']['enabled']:
        transforms.append(A.VerticalFlip(p=config['augmentations']['VerticalFlip']['p']))
    if config['augmentations']['GaussNoise']['enabled']:
        transforms.append(A.GaussNoise(
            var_limit=config['augmentations']['GaussNoise']['vara_limit'],
            p=config['augmentations']['GaussNoise']['p']
        ))
    if config['augmentations']['Rotate']['enabled']:
        transforms.append(A.Rotate(
            limit=config['augmentations']['Rotate']['limit'],
            p=config['augmentations']['Rotate']['p']
        ))
    
    return A.Compose(transforms, bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))

def apply_augmentation(image, bboxes, category_ids, augmentation_pipeline):
    augmented = augmentation_pipeline(image=image, bboxes=bboxes, category_ids=category_ids)
    return augmented['image'], augmented['bboxes']

def parse_bboxes(root):
    bboxes = []
    category_ids = []
    for member in root.findall('object'):
        bndbox = member.find('bndbox')
        bboxes.append([
            int(bndbox.find('xmin').text),
            int(bndbox.find('ymin').text),
            int(bndbox.find('xmax').text),
            int(bndbox.find('ymax').text)
        ])
        category_ids.append(member.find('name').text)
    return bboxes, category_ids

def update_bboxes(root, bboxes):
    for member, bbox in zip(root.findall('object'), bboxes):
        bndbox = member.find('bndbox')
        bndbox.find('xmin').text = str(int(bbox[0]))
        bndbox.find('ymin').text = str(int(bbox[1]))
        bndbox.find('xmax').text = str(int(bbox[2]))
        bndbox.find('ymax').text = str(int(bbox[3]))

def augment_images_and_xmls(config):
    image_paths = glob.glob(config['image_path'])
    xml_paths = glob.glob(config['xml_path'])
    output_image_dir = os.path.join(config['output_path'], 'images')
    output_xml_dir = os.path.join(config['output_path'], 'xmls')
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_xml_dir, exist_ok=True)

    augmentation_pipeline = create_augmentation_pipeline(config)

    for image_path, xml_path in zip(image_paths, xml_paths):
        image, root = load_image_and_xml(image_path, xml_path)
        bboxes, category_ids = parse_bboxes(root)
        
        for i in range(config['num_augmentations']):
            augmented_image, augmented_bboxes = apply_augmentation(image, bboxes, category_ids, augmentation_pipeline)
            update_bboxes(root, augmented_bboxes)

            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_image_path = os.path.join(output_image_dir, f"{base_name}_aug_{i}.PNG")
            output_xml_path = os.path.join(output_xml_dir, f"{base_name}_aug_{i}.xml")

            save_augmented_image_and_xml(augmented_image, root, output_image_path, output_xml_path)
        
        print(f"Image {os.path.basename(image_path)} augmented and created {config['num_augmentations']} augmentations.")

if __name__ == '__main__':
    config = {
        "image_path": "./train/images/*.png",
        "xml_path": "./train/xmls/*.xml",
        "output_path": "./aug/",
        "num_augmentations": 5,
        "augmentations": {
            "RandomBrightnessContrast": {"enabled": True, "brightness_limit": 0.1, "contrast_limit": 0.1, "p": 1},
            "Sharpen": {"enabled": True, "p": 0.2},
            "MotionBlur": {"enabled": True, "p": 0.2},
            "HorizontalFlip": {"enabled": True, "p": 0.3},
            "VerticalFlip": {"enabled": True, "p": 0.3},
            "GaussNoise": {"enabled": True, "vara_limit": (10.0, 50.0), "p": 0.2},
            "Rotate": {"enabled": True, "limit": (-5, 5), "p": 0.1},
        }
    }

    augment_images_and_xmls(config)
    print("Data augmentation completed successfully.")