from yolov3.yolov4 import Create_Yolo
from yolov3.utils import detect_image
from yolov3.configs import YOLO_INPUT_SIZE, TRAIN_CLASSES, TRAIN_MODEL_NAME



image_path= "./IMAGES/kite.jpg"
video_path= "./IMAGES/test.mp4"

yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES)
yolo.load_weights(f"./checkpoints/{TRAIN_MODEL_NAME}")

detect_image(yolo, image_path, "./IMAGES/predict.jpg", input_size=YOLO_INPUT_SIZE, show=True, rectangle_colors=(255,0,0))

#detect_video(yolo, video_path, "", input_size=YOLO_INPUT_SIZE, show=False, rectangle_colors=(255,0,0))
#detect_realtime(yolo, '', input_size=YOLO_INPUT_SIZE, show=True, rectangle_colors=(255, 0, 0))

#detect_video_realtime_mp(video_path, "Output.mp4", input_size=YOLO_INPUT_SIZE, show=False, rectangle_colors=(255,0,0), realtime=False)
