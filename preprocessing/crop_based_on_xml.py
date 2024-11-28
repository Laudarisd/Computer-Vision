from collections import defaultdict
import csv
import errno
from PIL import Image
import xml.etree.ElementTree as ET
import os
import glob
from concurrent.futures import ThreadPoolExecutor
import logging

class XMLParser:
    def __init__(self, img_folder, dst, xmls):
        self.img_folder = img_folder
        self.dst = dst
        self.xmls = xmls
        self.seed_arr = []
        self.img_class_counts = defaultdict(lambda: defaultdict(int))
        
        # Set up logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Create destination directory at initialization
        self.check_folder_exists(dst)

    def check_folder_exists(self, path):
        if not os.path.exists(path):
            try:
                os.makedirs(path)
                self.logger.info(f'Created directory: {path}')
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

    def parse_xml_files(self):
        xml_files = glob.glob(os.path.join(self.xmls, '*.xml'))
        # Process XML files in parallel
        with ThreadPoolExecutor() as executor:
            executor.map(self.parse_xml_file, xml_files)

    def parse_xml_file(self, xml_file):
        try:
            root = ET.parse(xml_file).getroot()
            filename = root.find('filename').text
            size = root.find('size')
            width = size.find('width').text
            height = size.find('height').text

            objects = []
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                self.img_class_counts[filename][class_name] += 1
                
                bndbox = obj.find('bndbox')
                xmin = bndbox.find('xmin').text
                ymin = bndbox.find('ymin').text
                xmax = bndbox.find('xmax').text
                ymax = bndbox.find('ymax').text
                
                objects.append([filename, width, height, class_name, xmin, ymin, xmax, ymax])
            
            # Thread-safe append to seed_arr
            self.seed_arr.extend(objects)
            
        except Exception as e:
            self.logger.error(f"Error processing {xml_file}: {str(e)}")

    def save_results_to_csv(self, csv_file):
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['img_name', 'class', 'total'])
            for img_name, class_counts in self.img_class_counts.items():
                for class_name, count in class_counts.items():
                    writer.writerow([img_name, class_name, count])

    def process_image(self, line):
        try:
            filename, width, height, class_name, xmin, ymin, xmax, ymax = line
            load_img_path = os.path.join(self.img_folder, filename)
            
            if not os.path.exists(load_img_path):
                self.logger.warning(f"Image file not found: {load_img_path}")
                return

            save_class_path = os.path.join(self.dst, class_name)
            self.check_folder_exists(save_class_path)
            
            filename_without_ext = os.path.splitext(filename)[0]
            
            # Calculate centered points
            x_center = format((int(xmin) + int(xmax)) / 2, ".2f")
            y_center = format((int(ymin) + int(ymax)) / 2, ".2f")
            
            save_img_path = os.path.join(
                save_class_path, 
                f"{id(line)}_{filename_without_ext}_{x_center}_{y_center}.png"
            )

            # Process image
            with Image.open(load_img_path) as img:
                crop_img = img.crop((int(xmin), int(ymin), int(xmax), int(ymax)))
                
                # Calculate new size while preserving aspect ratio
                scale_factor = 10
                original_width = int(xmax) - int(xmin)
                original_height = int(ymax) - int(ymin)
                
                if original_width > original_height:
                    new_width = original_width * scale_factor
                    new_height = original_height * scale_factor
                else:
                    new_width = original_width * scale_factor
                    new_height = original_height * scale_factor

                # Resize and save
                crop_img = crop_img.resize((new_width, new_height))
                crop_img.save(save_img_path, 'PNG', optimize=True)
                
            self.logger.info(f"Saved {save_img_path}")
            
        except Exception as e:
            self.logger.error(f"Error processing image {filename}: {str(e)}")

    def save_images(self):
        self.seed_arr.sort()
        # Process images in parallel
        with ThreadPoolExecutor() as executor:
            executor.map(self.process_image, self.seed_arr)

if __name__ == '__main__':
    img_folder = './images'
    dst = './cropped'
    xmls = './xmls'
    #csv_file = './results.csv'

    parser = XMLParser(img_folder, dst, xmls)
    parser.parse_xml_files()
    #parser.save_results_to_csv(csv_file)
    parser.save_images()
