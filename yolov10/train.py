import os
from ultralytics import YOLO
os.environ['WANDB_MODE'] = 'disabled'


def train_model(data_yaml_path):
    try:
        #model = YOLO('yolov8l.pt')  
        # To train yolo v10 
        model = YOLO('yolov10l.pt')
        model.train(
            data=data_yaml_path,
            epochs=300,
            batch=4,
            imgsz=1024,
            device='1',
            mixup=0.3,
            copy_paste=0.5,
            label_smoothing=0.3,
            degrees=10,
            multi_scale=True,
            cos_lr=True,
        )
        
    except Exception as e:
        print(f"An error occurred during training with parameters : {e}")

