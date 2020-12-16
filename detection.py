import os
if 'CUDA_VISIBLE_DEVICES' not in os.environ.keys():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
if 'TF_FORCE_GPU_ALLOW_GROWTH' not in os.environ.keys():
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'
print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
print(f"TF_FORCE_GPU_ALLOW_GROWTH: {os.environ['TF_FORCE_GPU_ALLOW_GROWTH']}")

from yolov3.yolov4 import Create_Yolo
from yolov3.utils import detect_image
from yolov3.configs import YOLO_INPUT_SIZE, TRAIN_CLASSES, TRAIN_MODEL_NAME


# image_path = os.path.join("IMAGES", "kite.jpg")
image_path = [
  os.path.join("dataset", "p1", "JPEGImages", "frame_004500.PNG"),
  os.path.join("dataset", "p1", "JPEGImages", "frame_000480.PNG"),
  os.path.join("dataset", "p1", "JPEGImages", "frame_001330.PNG"),
  os.path.join("dataset", "p1", "JPEGImages", "frame_001730.PNG"),
]
video_path = os.path.join("IMAGES", "test.mp4")
output_path = os.path.join("IMAGES", "predict")

yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES)
yolo.load_weights(f"./checkpoints/{TRAIN_MODEL_NAME}")

# output_path = None

detect_image(yolo, image_path, output_path, input_size=YOLO_INPUT_SIZE, show=True, return_images=False, rectangle_colors=(255,0,0))
