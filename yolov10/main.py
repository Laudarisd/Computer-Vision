import os
from src.xml_to_yolo import VOCToYOLO
from train import train_model
os.environ['WANDB_MODE'] = 'disabled'



def main():
    # Step 1: Convert XML files to YOLO format and move images
    raw_data_path = './raw_data'
    output_dir = './custom_dataset'
    if os.path.isdir(output_dir) is False:
        converter = VOCToYOLO(raw_data_path, output_dir)
        converter.process_folder()

    # Step 2: Import yaml file
    data_yaml_path = os.path.join(output_dir, 'custom.yaml')

    # Step 3: Select a set of parameters to train with
    
    train_model(data_yaml_path)

if __name__ == '__main__':
    main()
