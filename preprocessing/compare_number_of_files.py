import os
import shutil

#Compare files name and number in xmls, and print which folder has more files and what files
def comparedfiles(images_dir, xmls_dir):
    #parse image files and remove extension
    image_files = []
    for file in os.listdir(images_dir):
        if file.endswith(('.png', '.jpg', '.PNG', '.JPG')):
            image_files.append(os.path.splitext(file)[0])
    #parse xml files and remove extension
    xml_files = []
    for file in os.listdir(xmls_dir):
        if file.endswith('.xml'):
            xml_files.append(os.path.splitext(file)[0])

    #compare files
    if len(image_files) > len(xml_files):
        print(f"More image files than xml files: {len(image_files)} > {len(xml_files)}")
        print("Extra image files:")
        for file in image_files:
            if file not in xml_files:
                print(file)
    elif len(image_files) < len(xml_files):
        print(f"More xml files than image files: {len(xml_files)} > {len(image_files)}")
        print("Extra xml files:")
        for file in xml_files:
            if file not in image_files:
                print(file)
    else:
        print(f"Number of image files and xml files are equal: {len(image_files)}")

if __name__ == '__main__':
    images_dir = './raw_data-v2/old_data/images'
    xmls_dir = './raw_data-v2/old_data/xmls'
    comparedfiles(images_dir, xmls_dir)