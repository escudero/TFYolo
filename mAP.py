import os
if 'CUDA_VISIBLE_DEVICES' not in os.environ.keys():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
if 'TF_FORCE_GPU_ALLOW_GROWTH' not in os.environ.keys():
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'
print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
print(f"TF_FORCE_GPU_ALLOW_GROWTH: {os.environ['TF_FORCE_GPU_ALLOW_GROWTH']}")

from yolov3.dataset import Dataset
from yolov3.utils import detect_image
from yolov3.yolov4 import Create_Yolo
from yolov3.configs import YOLO_INPUT_SIZE, YOLO_CLASSES, TRAIN_CHECKPOINTS_FOLDER, TRAIN_MODEL_NAME
from map_boxes import mean_average_precision_for_boxes
import numpy as np
import pandas as pd
# from tqdm import tqdm
# from tqdm.notebook import tqdm


def to_ann(ann):
    coords = np.array([c.split(',') for c in ann[1]]).astype(float)
    sizes = ann[2].shape
    coords[:,[0,2]] /= sizes[1]
    coords[:,[1,3]] /= sizes[0]
    coords = np.concatenate([len(coords) * [[ann[0]]], coords[:,4][..., np.newaxis], coords[:,0:4]], axis=-1)
    df = pd.DataFrame(coords, columns=['ImageID','LabelName','XMin','YMin','XMax','YMax'])
    df = df[['ImageID','LabelName','XMin','XMax','YMin','YMax']]
    df[['XMin','XMax','YMin','YMax']] = df[['XMin','XMax','YMin','YMax']].astype(float)
    return df.values

def to_det(image_id, sizes, det):
    coords = det[0]
    if len(coords) == 0:
        return np.zeros([0, 7])
    coords[:,[0,2]] /= sizes[1]
    coords[:,[1,3]] /= sizes[0]
    coords = np.concatenate([len(coords) * [[image_id]], coords[:,5][..., np.newaxis], coords[:,4][..., np.newaxis], coords[:,0:4]], axis=-1)
    df = pd.DataFrame(coords, columns=['ImageID','LabelName','Conf','XMin','YMin','XMax','YMax'])
    df = df[['ImageID','LabelName','Conf','XMin','XMax','YMin','YMax']]
    df[['Conf','XMin','XMax','YMin','YMax']] = df[['Conf','XMin','XMax','YMin','YMax']].astype(float)
    return df.values

def getmAP(model, dataset, iou_threshold, method):
    ann_list = []
    det_list = []
    # for ann_dataset in tqdm(dataset.annotations, desc="Dataset Loop", leave=True):
    for ann_dataset in dataset.annotations[:100]:
        image_path = ann_dataset[2]
        pred = detect_image(model, image_path, iou_threshold=iou_threshold, method=method, output_path=None, input_size=YOLO_INPUT_SIZE, show=False, return_images=False, class_names=YOLO_CLASSES)
        ann_list.append(to_ann(ann_dataset))
        det_list.append(to_det(ann_dataset[0], ann_dataset[2].shape, pred))

    ann_list = np.concatenate(ann_list)
    det_list = np.concatenate(det_list)

    mean_ap, average_precisions = mean_average_precision_for_boxes(ann_list, det_list, verbose=False)
    return mean_ap


dataset = Dataset('test', input_size=YOLO_INPUT_SIZE)

model = Create_Yolo(input_size=YOLO_INPUT_SIZE, class_names=YOLO_CLASSES)
model.load_weights(os.path.join(TRAIN_CHECKPOINTS_FOLDER, TRAIN_MODEL_NAME))


# for iou_threshold in tqdm(range(30, 100, 5), desc="IoU Loop", leave=True):
for iou_threshold in range(30, 100, 5):
    iou_threshold /= 100.0
    # for method in tqdm(["nms", "nmw", "wbf"], desc="Method Loop", leave=True): # ["nms", "soft_nms", "nmw", "wbf"]
    for method in ["nms", "nmw", "wbf"]: # ["nms", "soft_nms", "nmw", "wbf"]
        map = getmAP(model, dataset, iou_threshold, method)
        print(f'{method} | {iou_threshold} | {map}')

print(f'soft_nms | 0.5 | {getmAP(model, dataset, iou_threshold=0.5, method="soft_nms")}')
print(f'soft_nms | 0.9 | {getmAP(model, dataset, iou_threshold=0.9, method="soft_nms")}')


# print(f'wbf | 0.3 | {getmAP(model, dataset, iou_threshold=0.3, method="wbf")}')
# print(f'nms | 0.3 | {getmAP(model, dataset, iou_threshold=0.3, method="nms")}')