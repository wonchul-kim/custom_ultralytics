from ultralytics import YOLO, settings
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
settings.update({'wandb': False})


model = YOLO("/HDD/weights/yolov11/yolo11x.pt")

data={'path': '/DeepLearning/_athena_tests/datasets/rectangle2/split_dataset_unit',
        'train': 'train',
        'val': 'val',
        'names': {0: 'RING', 1: 'DUST', 2: 'SCRATCH', 3: 'FOLD', 4: 'DAMAGE', 5: 'LINE', 
                  6: 'BOLD', 7: 'BURR', 8: 'BUBBLE', 9: 'TIP', 10: 'REACT'}
    }

train_results = model.train(
    data=data,  
    epochs=2, 
    imgsz=1024, 
    device="0,1", 
    lrf=0.001,
    batch=4,
    
    label_format='labelme',
    roi_info=[[300, 300, 800, 800]],
)
