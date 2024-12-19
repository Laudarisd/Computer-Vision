import os
import shutil

#Compare files name and number in xmls, and print which folder has more files and what files
def comparedfiles(images_dir, xmls_dir, copy_exta_file):
    #parse image files and remove extension
    image_files = [] ## raw pdf
    for file in os.listdir(images_dir):
        if file.endswith(( 'pdf')):
            image_files.append(os.path.splitext(file)[0])
    #parse xml files and remove extension
    xml_files = [] # fine pdf
    for file in os.listdir(xmls_dir):
        if file.endswith('.pdf'):
            xml_files.append(os.path.splitext(file)[0])

    #compare files
    if len(image_files) > len(xml_files):
        print(f"More raw pdf files than fine pdf files: {len(image_files)} > {len(xml_files)}")
        print("Extra raw pdf files:")
        
        for file in image_files:
            if file not in xml_files:
                print(file)
                shutil.copy(f'{images_dir}/{file}.pdf', f'{copy_exta_file}/{file}.pdf')
    elif len(image_files) < len(xml_files):
        print(f"More fine pdf files than raw pdf files: {len(xml_files)} > {len(image_files)}")
        print("Extra fine pdf files:")
        for file in xml_files:
            if file not in image_files:
                print(file)
                shutil.copy(f'{xmls_dir}/{file}.pdf', f'{copy_exta_file}/{file}.pdf')

        
    else:
        print(f"Number of raw pdf files and fine pdf files are equal: {len(image_files)}")

if __name__ == '__main__':
    images_dir = './raw_data/combined'
    xmls_dir = './raw_data/cv_cyl'
    copy_exta_file = './raw_data/extra_files'
    os.makedirs(copy_exta_file, exist_ok=True)
    comparedfiles(images_dir, xmls_dir, copy_exta_file)
