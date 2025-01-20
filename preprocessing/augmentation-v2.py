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
    filename_elem = root.find('filename')
    if filename_elem is not None:
        filename_elem.text = os.path.basename(new_image_path)
    else:
        filename_elem = ET.SubElement(root, 'filename')
        filename_elem.text = os.path.basename(new_image_path)

    path_elem = root.find('path')
    if path_elem is not None:
        path_elem.text = os.path.abspath(new_image_path)
    else:
        path_elem = ET.SubElement(root, 'path')
        path_elem.text = os.path.abspath(new_image_path)

def save_augmented_image_and_xml(image, root, output_image_path, output_xml_path):
    # Update size information in XML
    size = root.find('size')
    if size is None:
        size = ET.SubElement(root, 'size')
        ET.SubElement(size, 'width').text = str(image.shape[1])
        ET.SubElement(size, 'height').text = str(image.shape[0])
        ET.SubElement(size, 'depth').text = str(image.shape[2])
    else:
        size.find('width').text = str(image.shape[1])
        size.find('height').text = str(image.shape[0])
        size.find('depth').text = str(image.shape[2])

    update_xml_metadata(root, output_image_path)
    cv2.imwrite(output_image_path, image)
    tree = ET.ElementTree(root)
    tree.write(output_xml_path)

def create_augmentation_pipeline(config, height, width):
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
            var_limit=config['augmentations']['GaussNoise']['var_limit'],
            p=config['augmentations']['GaussNoise']['p']
        ))
    if config['augmentations']['Rotate']['enabled']:
        # Update Rotate transform to handle bounding boxes properly
        transforms.append(A.Rotate(
            limit=config['augmentations']['Rotate']['limit'],
            p=config['augmentations']['Rotate']['p'],
            border_mode=cv2.BORDER_CONSTANT,
            value=None,
            crop_border=True  # This ensures rotated bounding boxes stay within image bounds
        ))

    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['category_ids'],
            min_visibility=0.3  # Only keep bboxes that are at least 30% visible after transformation
        )
    )

def apply_augmentation(image, bboxes, category_ids, augmentation_pipeline):
    try:
        augmented = augmentation_pipeline(
            image=image,
            bboxes=bboxes,
            category_ids=category_ids
        )
        return augmented['image'], augmented['bboxes'], True
    except Exception as e:
        print(f"Warning: Augmentation failed - {str(e)}")
        return image, bboxes, False

def parse_bboxes(root):
    bboxes = []
    category_ids = []
    for member in root.findall('object'):
        bndbox = member.find('bndbox')
        bbox = [
            float(bndbox.find('xmin').text),
            float(bndbox.find('ymin').text),
            float(bndbox.find('xmax').text),
            float(bndbox.find('ymax').text)
        ]
        # Ensure bbox coordinates are within image bounds
        bbox = [max(0, min(coord, 99999)) for coord in bbox]
        bboxes.append(bbox)
        category_ids.append(member.find('name').text)
    return bboxes, category_ids

def update_bboxes(root, bboxes, category_ids):
    # Remove existing objects
    for obj in root.findall('object'):
        root.remove(obj)
    
    # Add updated objects
    for bbox, category_id in zip(bboxes, category_ids):
        obj = ET.SubElement(root, 'object')
        ET.SubElement(obj, 'name').text = category_id
        ET.SubElement(obj, 'pose').text = 'Unspecified'
        ET.SubElement(obj, 'truncated').text = '0'
        ET.SubElement(obj, 'difficult').text = '0'
        
        bndbox = ET.SubElement(obj, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(int(round(bbox[0])))
        ET.SubElement(bndbox, 'ymin').text = str(int(round(bbox[1])))
        ET.SubElement(bndbox, 'xmax').text = str(int(round(bbox[2])))
        ET.SubElement(bndbox, 'ymax').text = str(int(round(bbox[3])))

def augment_images_and_xmls(config):
    image_paths = glob.glob(config['image_path'])
    xml_paths = glob.glob(config['xml_path'])
    output_image_dir = os.path.join(config['output_path'], 'images')
    output_xml_dir = os.path.join(config['output_path'], 'xmls')
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_xml_dir, exist_ok=True)

    for image_path, xml_path in zip(image_paths, xml_paths):
        image, root = load_image_and_xml(image_path, xml_path)
        height, width = image.shape[:2]
        
        # Create augmentation pipeline with image dimensions
        augmentation_pipeline = create_augmentation_pipeline(config, height, width)
        
        original_bboxes, category_ids = parse_bboxes(root)
        
        for i in range(config['num_augmentations']):
            augmented_image, augmented_bboxes, success = apply_augmentation(
                image, original_bboxes, category_ids, augmentation_pipeline
            )
            
            if success:
                # Create a new XML root for each augmentation
                new_root = ET.Element('annotation')
                for child in root:
                    if child.tag not in ['object']:  # Copy everything except objects
                        new_root.append(ET.ElementTree(child).getroot())
                
                # Update bounding boxes in the new XML
                update_bboxes(new_root, augmented_bboxes, category_ids)

                base_name = os.path.splitext(os.path.basename(image_path))[0]
                output_image_path = os.path.join(output_image_dir, f"{base_name}_aug_{i}.png")
                output_xml_path = os.path.join(output_xml_dir, f"{base_name}_aug_{i}.xml")

                save_augmented_image_and_xml(augmented_image, new_root, output_image_path, output_xml_path)
                print(f"Created augmentation {i+1} for {os.path.basename(image_path)}")

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
            "GaussNoise": {"enabled": True, "var_limit": (10.0, 20.0), "p": 0.2},
            "Rotate": {"enabled": True, "limit": (-5, 5), "p": 0.1},
        }
    }

    augment_images_and_xmls(config)
    print("Data augmentation completed successfully.")