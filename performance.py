import os
if 'CUDA_VISIBLE_DEVICES' not in os.environ.keys():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
if 'TF_FORCE_GPU_ALLOW_GROWTH' not in os.environ.keys():
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'
print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
print(f"TF_FORCE_GPU_ALLOW_GROWTH: {os.environ['TF_FORCE_GPU_ALLOW_GROWTH']}")

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
print(f"Quantidade de GPUs: {len(gpus)}")

from yolov3.yolov4 import Create_Yolo
from yolov3.utils import detect_image, pre_processing, predict, post_processing
from yolov3.configs import YOLO_INPUT_SIZE, YOLO_CLASSES, TRAIN_MODEL_NAME, TRAIN_CHECKPOINTS_FOLDER
import numpy as np
from datetime import datetime

model = Create_Yolo(input_size=YOLO_INPUT_SIZE, class_names=YOLO_CLASSES)
model.load_weights(os.path.join(TRAIN_CHECKPOINTS_FOLDER, TRAIN_MODEL_NAME))

image_path = os.path.join('IMAGES', 'futebol01.png')

# Executando pelo menso 1x antes das performance para carregar todas as dependências
detect_image(model, [image_path], output_path=None, input_size=YOLO_INPUT_SIZE, show=False, return_images=False, class_names=YOLO_CLASSES)

for t in [1, 5, 10, 60]:
    images_path = [image_path] * 30 * t

    print(f'# Testando de {t} / Pre-processamento', end=' : ')
    dt1 = datetime.now()
    images_data, original_images = pre_processing(images_path, input_size=YOLO_INPUT_SIZE)
    print(f'Tempo de {t}: {(datetime.now() - dt1).total_seconds()}')

    print(f'# Testando de {t} / Predict', end=' : ')
    dt1 = datetime.now()
    pred_bboxs = predict(model, images_data)
    print(f'Tempo de {t}: {(datetime.now() - dt1).total_seconds()}')
    
    print(f'# Testando de {t} / Pos-processamento', end=' : ')
    dt1 = datetime.now()
    post_processing(pred_bboxs, original_images, input_size=YOLO_INPUT_SIZE)
    print(f'Tempo de {t}: {(datetime.now() - dt1).total_seconds()}')
  
    print(f'# Testando de {t} / Total', end=' : ')
    dt1 = datetime.now()
    detect_image(model, images_path, output_path=None, input_size=YOLO_INPUT_SIZE, show=False, return_images=False, class_names=YOLO_CLASSES)
    print(f'Tempo de {t}: {(datetime.now() - dt1).total_seconds()}')
  