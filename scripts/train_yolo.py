from ultralytics import YOLO, settings
settings.update({'wandb': False})


model = YOLO("/HDD/weights/yolov11/yolo11x.pt")

train_results = model.train(
    data="/HDD/_projects/github/custom_ultralytics/scripts/cfgs/test.yaml",  
    epochs=300, 
    imgsz=1024, 
    device="0,1,2,3", 
    lrf=0.001,
    batch=16,
)

