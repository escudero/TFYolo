import os
if 'CUDA_VISIBLE_DEVICES' not in os.environ.keys():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
if 'TF_FORCE_GPU_ALLOW_GROWTH' not in os.environ.keys():
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'
print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
print(f"TF_FORCE_GPU_ALLOW_GROWTH: {os.environ['TF_FORCE_GPU_ALLOW_GROWTH']}")

from yolov3.yolov4 import Create_Yolo
from yolov3.utils import detect_image, detect_video
from yolov3.configs import YOLO_INPUT_SIZE, TRAIN_MODEL_NAME, YOLO_CLASSES, TRAIN_CHECKPOINTS_FOLDER
from datetime import datetime


# image_path = os.path.join("IMAGES", "kite.jpg")
image_path = [
  os.path.join("IMAGES", "futebol01.png"),
  os.path.join("IMAGES", "futebol02.png"),
  os.path.join("IMAGES", "futebol01.png"),
  os.path.join("IMAGES", "futebol02.png"),
]
video_path = os.path.join("IMAGES", "futebol.mp4")
output_path = os.path.join("IMAGES", "predict")

model = Create_Yolo(input_size=YOLO_INPUT_SIZE, class_names=YOLO_CLASSES)
model.load_weights(os.path.join(TRAIN_CHECKPOINTS_FOLDER, TRAIN_MODEL_NAME))

# output_path = None

dt1 = datetime.now()
detect_image(model, image_path, output_path, input_size=YOLO_INPUT_SIZE, show=False, return_images=False, class_names=YOLO_CLASSES, rectangle_colors=(255,0,0))
# detect_video(model, video_path, output_path, input_size=YOLO_INPUT_SIZE, show=True, class_names=YOLO_CLASSES, chunksize=10, score_threshold=0.3, iou_threshold=0.45, rectangle_colors='')
dt2 = datetime.now()
print(f'Processing time: {dt2 - dt1}')
