from ultralytics import YOLO, settings
settings.update({'wandb': False})

format = 'onnx'
batch = 1
opset = 13
workspace = 8
device = 'cuda'
weights_file = "/DeepLearning/etc/_athena_tests/recipes/agent/detection/ultralytics/train_unit/yolov11/outputs/DETECTION/2024_12_23_11_34_21/train/weights/last.pt"
output_dir = f'/HDD/etc/outputs/ultralytics'

model = YOLO(weights_file)
# model.add_callback('on_export_start', on_export_start)
model.export(format=format, batch=batch, opset=opset, 
             workspace=workspace, device=device)