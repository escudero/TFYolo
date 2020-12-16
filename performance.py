import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
print(f"Quantidade de GPUs: {len(gpus)}")

import cv2

from yolov3.yolov4 import Create_Yolo
from yolov3.utils import detect_image
from yolov3.configs import YOLO_INPUT_SIZE, YOLO_CLASSES, TRAIN_MODEL_NAME, YOLO_FRAMEWORK

model = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=YOLO_CLASSES)
model.load_weights(f"./checkpoints/{TRAIN_MODEL_NAME}")

import numpy as np
import tensorflow as tf
from yolov3.utils import image_resize, postprocess_boxes, nms

def pre_processing(image_path, input_size=YOLO_INPUT_SIZE):
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    image_data = image_resize(np.copy(original_image), [input_size, input_size])
    image_data = image_data[np.newaxis, ...].astype(np.float32)
    return image_data, original_image

def predict(model, image_data):
    if YOLO_FRAMEWORK == "tf":
        return model.predict(image_data)
    elif YOLO_FRAMEWORK == "trt":
        batched_input = tf.constant(image_data)
        result = model(batched_input)
        pred_bbox = []
        for key, value in result.items():
            value = value.numpy()
            pred_bbox.append(value)
        return pred_bbox
    return None

def post_processing(pred_bbox, original_image, input_size=YOLO_INPUT_SIZE, score_threshold=0.3, iou_threshold=0.45):
    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = tf.concat(pred_bbox, axis=0)
    
    bboxes = postprocess_boxes(pred_bbox, original_image, input_size, score_threshold)
    bboxes = nms(bboxes, iou_threshold, method='nms')
    return bboxes 

def detect_image_only_boundingbox(model, image_path, input_size=YOLO_INPUT_SIZE, score_threshold=0.3, iou_threshold=0.45):
    image_data, original_image = pre_processing(image_path, input_size)
    pred_bbox = predict(model, image_data)
    bboxes = post_processing(pred_bbox, original_image, input_size=YOLO_INPUT_SIZE, score_threshold=0.3, iou_threshold=0.45)
    return bboxes

from datetime import datetime
image_path ="dataset/JPEGImages/frame_004500.PNG"

detect_image_only_boundingbox(model, image_path, input_size=YOLO_INPUT_SIZE, score_threshold=0.3, iou_threshold=0.45)
image_data, original_image = pre_processing(image_path, input_size=YOLO_INPUT_SIZE)
pred_bbox = predict(model, image_data)

for t in [1, 5, 10, 60]:
    print(f'# Testando de {t} / Pre-processamento', end=' : ')
    dt1 = datetime.now()
    for _ in range(30 * t):
        pre_processing(image_path, input_size=YOLO_INPUT_SIZE)
    print(f'Tempo de {t}: {(datetime.now() - dt1).total_seconds()}')

    print(f'# Testando de {t} / Predict', end=' : ')
    dt1 = datetime.now()
    for _ in range(30 * t):
        predict(model, image_data)
    print(f'Tempo de {t}: {(datetime.now() - dt1).total_seconds()}')
    
    print(f'# Testando de {t} / Pos-processamento', end=' : ')
    dt1 = datetime.now()
    for _ in range(30 * t):
        post_processing(pred_bbox, original_image, input_size=YOLO_INPUT_SIZE, score_threshold=0.3, iou_threshold=0.45)
    print(f'Tempo de {t}: {(datetime.now() - dt1).total_seconds()}')
  
    print(f'# Testando de {t} / Total', end=' : ')
    dt1 = datetime.now()
    for _ in range(30 * t):
        detect_image_only_boundingbox(model, image_path, input_size=YOLO_INPUT_SIZE, score_threshold=0.3, iou_threshold=0.45)
    print(f'Tempo de {t}: {(datetime.now() - dt1).total_seconds()}')
  