from ultralytics import YOLO, settings
settings.update({'wandb': False})


model = YOLO("/HDD/weights/yolov11/yolo11x.pt")

train_results = model.train(
    data="/HDD/_projects/github/custom_ultralytics/scripts/cfgs/unit.yaml",  
    epochs=300, 
    imgsz=1024, 
    device="0,1", 
    lrf=0.001,
    batch=4,
    optimizer="SGD"
)