import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
print(f"Quantidade de GPUs: {len(gpus)}")

import cv2

from yolov3.yolov4 import Create_Yolo
from yolov3.utils import detect_image
from yolov3.configs import YOLO_INPUT_SIZE, TRAIN_CLASSES, TRAIN_MODEL_NAME, YOLO_FRAMEWORK

yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES)
yolo.load_weights(f"./checkpoints/{TRAIN_MODEL_NAME}")

import numpy as np
import tensorflow as tf
from yolov3.utils import image_preprocess, postprocess_boxes, nms

def detect_image_only_boundingbox(model, image_path, input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES, score_threshold=0.3, iou_threshold=0.45):
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    image_data = image_preprocess(np.copy(original_image), [input_size, input_size])
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    if YOLO_FRAMEWORK == "tf":
        pred_bbox = model.predict(image_data)
    elif YOLO_FRAMEWORK == "trt":
        batched_input = tf.constant(image_data)
        result = model(batched_input)
        pred_bbox = []
        for key, value in result.items():
            value = value.numpy()
            pred_bbox.append(value)

    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = tf.concat(pred_bbox, axis=0)
    
    bboxes = postprocess_boxes(pred_bbox, original_image, input_size, score_threshold)
    bboxes = nms(bboxes, iou_threshold, method='nms')

    return bboxes

from datetime import datetime
image_path = './dataset/frame_004500.PNG'

for t in [1, 5, 10, 60]:
  print(f'\n\nTestando de {t}')
  dt1 = datetime.now()
  for _ in range(30 * t):
      detect_image_only_boundingbox(yolo, image_path, input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES, score_threshold=0.3, iou_threshold=0.45)
  dt2 = datetime.now()
  print(f'Tempo de {t}: {dt2 - dt1}\n')