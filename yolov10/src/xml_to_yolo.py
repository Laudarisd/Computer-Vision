
from xml.dom import minidom
import os
import glob
import yaml
import shutil
from tqdm import tqdm
import platform

def convert_coordinates(size, box):
    """
    Convert VOC box coordinates to YOLO format.

    Parameters:
    - size: Tuple of (width, height) of the image.
    - box: Tuple of (xmin, xmax, ymin, ymax) for the bounding box.

    Returns:
    - Tuple of (x, y, w, h) in YOLO format.
    """
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (x * dw, y * dh, w * dw, h * dh)

class VOCToYOLO:
    def __init__(self, raw_data_path, output_dir='./custom_data'):
        """
        Initialize the converter.

        Parameters:
        - raw_data_path: Path to the raw VOC data.
        - output_dir: Path where the converted YOLO data should be stored.
        """
        self.raw_data_path = raw_data_path
        self.output_dir = output_dir if os.path.isabs(output_dir) else os.path.join(os.getcwd(), output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        self.lut = self.collect_classes()
        self.supported_image_formats = ['jpg', 'jpeg', 'PNG', 'png']

    def collect_classes(self):
        """
        Collect unique classes from VOC annotation files.

        Returns:
        - Dictionary mapping class names to indices.
        """
        unique_classes = set()
        for split in ['train', 'test', 'valid']:
            xml_folder = os.path.join(self.raw_data_path, split, 'xmls')
            for fname in glob.iglob(os.path.join(xml_folder, '*.xml')):
                try:
                    xmldoc = minidom.parse(fname)
                    itemlist = xmldoc.getElementsByTagName('object')
                    for item in itemlist:
                        classid = item.getElementsByTagName('name')[0].firstChild.data
                        unique_classes.add(classid)
                except Exception as e:
                    print(f"Error processing file {fname}: {e}")
        return {classname: index for index, classname in enumerate(sorted(unique_classes))}

    def convert_xml2yolo(self, split):
        """
        Convert VOC XML annotations to YOLO format for a given data split.

        Parameters:
        - split: The data split (train, test, or valid).
        """
        xml_folder = os.path.join(self.raw_data_path, split, 'xmls')
        image_folder = os.path.join(self.raw_data_path, split, 'images')
        output_folder = os.path.join(self.output_dir, split, 'labels')
        image_output_folder = os.path.join(self.output_dir, split, 'images')

        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(image_output_folder, exist_ok=True)

        for fname in tqdm(glob.iglob(os.path.join(xml_folder, '*.xml')), desc=f"Processing {split} data", unit='files'):
            try:
                xmldoc = minidom.parse(fname)
                fname_out = os.path.join(output_folder, os.path.basename(fname)[:-4] + '.txt')
                with open(fname_out, "w") as f:
                    itemlist = xmldoc.getElementsByTagName('object')
                    size = xmldoc.getElementsByTagName('size')[0]
                    width, height = [int(size.getElementsByTagName(dim)[0].firstChild.data) for dim in ('width', 'height')]
                    for item in itemlist:
                        classid = item.getElementsByTagName('name')[0].firstChild.data
                        label_str = str(self.lut[classid])
                        xmin, ymin, xmax, ymax = [float(item.getElementsByTagName('bndbox')[0].getElementsByTagName(coord)[0].firstChild.data) for coord in ('xmin', 'ymin', 'xmax', 'ymax')]
                        bb = convert_coordinates((width, height), (xmin, xmax, ymin, ymax))
                        f.write(label_str + " " + " ".join([("%.8f" % a) for a in bb]) + '\n')
                
                # Attempt to find and move the corresponding image file
                image_base = os.path.basename(fname)[:-4]
                image_found = False
                for ext in self.supported_image_formats:
                    src_image = os.path.join(image_folder, f"{image_base}.{ext}")
                    if os.path.isfile(src_image):
                        dst_image = os.path.join(image_output_folder, f"{image_base}.{ext}")
                        shutil.copy(src_image, dst_image)
                        image_found = True
                        break
                if not image_found:
                    print(f"Warning: No supported image file found for {fname}")
            except Exception as e:
                print(f"Error processing file {fname}: {e}")

    def generate_yaml(self):
        """
        Generate a YAML file with dataset paths and class information.
        """
        data = {
            'train': os.path.join(self.output_dir, 'train'),
            'val': os.path.join(self.output_dir, 'valid'),
            'test': os.path.join(self.output_dir, 'test'),
            'nc': len(self.lut),
            'names': {str(v): k for k, v in self.lut.items()}
        }

        yaml_file = os.path.join(self.output_dir, 'custom.yaml')
        try:
            with open(yaml_file, 'w') as f:
                yaml.safe_dump(data, f, sort_keys=False)
            print(f"\nYAML file generated: {yaml_file}")
        except Exception as e:
            print(f"Error generating YAML file: {e}")

    def generate_labeltxt(self):
        """
        Generate a text file mapping class names to indices.
        """
        labelmap_file = os.path.join(self.output_dir, 'labelmap.txt')
        try:
            with open(labelmap_file, 'w') as f:
                for class_name, class_id in self.lut.items():
                    f.write(f"{class_name} {class_id}\n")
            print(f"Generated labelmap file at {labelmap_file}")
        except Exception as e:
            print(f"Error generating label map file: {e}")

    def process_folder(self):
        """
        Process the entire dataset, converting annotations and organizing files.
        """
        for split in ['train', 'test', 'valid']:
            self.convert_xml2yolo(split)
        self.generate_yaml()
        self.generate_labeltxt()

# To use this script, you would instantiate VOCToYOLO with appropriate paths and call process_folder()
# converter = VOCT
