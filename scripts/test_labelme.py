from ultralytics import YOLO, settings
settings.update({'wandb': False})

import os
import os.path as osp
import glob
from tqdm import tqdm 
import numpy as np
import imgviz
# from vis_hbb import vis_hbb

weights_file = "/DeepLearning/etc/_athena_tests/recipes/agent/detection/ultralytics/train_unit/yolov11/outputs/DETECTION/2024_12_23_11_34_21/train/weights/last.pt"
input_dir = '/DeepLearning/_athena_tests/datasets/rectangle2/split_dataset_unit/val'
# json_dir = '/HDD/datasets/public/coco/test2017'
json_dir = '/DeepLearning/_athena_tests/datasets/rectangle2/split_dataset_unit/val'
output_dir = f'/HDD/etc/outputs/ultralytics'

if not osp.exists(output_dir):
    os.makedirs(output_dir)

compare_gt = False
iou_threshold = 0.9
conf_threshold = 0.1
line_width = 3
font_scale = 2
imgsz = 768
_classes = ['BOX']
input_img_ext = 'bmp'
output_img_ext = 'jpg'
output_img_size_ratio = 1


model = YOLO(weights_file)

if not osp.exists(output_dir):
    os.makedirs(output_dir)

img_files = glob.glob(osp.join(input_dir, f'*.{input_img_ext}'))

preds = {}
compare = {}
for img_file in tqdm(img_files):
    filename = osp.split(osp.splitext(img_file)[0])[-1]
    pred = model(img_file, save=False, imgsz=imgsz, iou=iou_threshold, conf=conf_threshold, verbose=False)[0]
    
    idx2class = pred.names
    orig_img = pred.orig_img
    orig_shape = pred.orig_shape
    boxes = pred.boxes
    cls = boxes.cls.tolist()
    conf = boxes.conf.tolist()
    xyxy = boxes.xyxy.tolist()
    
    idx2xyxys = {}
    for cls, conf, xyxy in zip(boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist()):
        if cls not in idx2xyxys.keys():
            idx2xyxys[cls] = {'bbox': [], 'confidence':[]}
                    
        idx2xyxys[cls]['bbox'].append([[int(np.round(xyxy[0])), int(np.round(xyxy[1]))], 
                                       [int(np.round(xyxy[2])), int(np.round(xyxy[3]))]])
        idx2xyxys[cls]['confidence'].append(conf)
        
    color_map = imgviz.label_colormap()[1:len(idx2class) + 1 + 1]
    
    # if compare_gt:
    #     _compare = vis_hbb(img_file, idx2xyxys, idx2class, output_dir, color_map, json_dir, 
    #                        compare_gt=compare_gt, iou_threshold=iou_threshold, line_width=line_width, font_scale=font_scale)
    #     _compare.update({"img_file": img_file})
    #     compare.update({filename: _compare})
    # else:
    #     vis_hbb(img_file, idx2xyxys, idx2class, output_dir, color_map, json_dir, 
    #             compare_gt=compare_gt, iou_threshold=iou_threshold, line_width=line_width, font_scale=font_scale)
            