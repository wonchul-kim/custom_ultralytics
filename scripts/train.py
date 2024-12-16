from ultralytics import YOLO, settings
settings.update({'wandb': False})


model = YOLO("/HDD/weights/yolov11/yolo11x.pt")

train_results = model.train(
    data="coco8.yaml",  # path to dataset YAML
    epochs=100,  # number of training epochs
    imgsz=80,  # training image size
    device="0",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
)

