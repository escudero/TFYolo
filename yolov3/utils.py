import tensorflow as tf
from yolov3.configs import (
    YOLO_TYPE,
    YOLO_V3_TINY_WEIGHTS,
    YOLO_V4_TINY_WEIGHTS,
    YOLO_V3_WEIGHTS,
    YOLO_V4_WEIGHTS,
    YOLO_INPUT_SIZE,
    YOLO_CLASSES,
    YOLO_FRAMEWORK,
    TRAIN_YOLO_TINY,
    TRAIN_MODEL_NAME,
    YOLO_METHOD_ENSEMBLEBOXES
)
from yolov3.yolov4 import Create_Yolo, read_class_names
import ensemble_boxes 
import cv2
import numpy as np
import random
import colorsys
import os

def load_yolo_weights(model, weights_file):
    tf.keras.backend.clear_session() # used to reset layer names
    # load Darknet original weights to TensorFlow model
    if YOLO_TYPE == "yolov3":
        range1 = 75 if not TRAIN_YOLO_TINY else 13
        range2 = [58, 66, 74] if not TRAIN_YOLO_TINY else [9, 12]
    if YOLO_TYPE == "yolov4":
        range1 = 110 if not TRAIN_YOLO_TINY else 21
        range2 = [93, 101, 109] if not TRAIN_YOLO_TINY else [17, 20]
    
    with open(weights_file, 'rb') as wf:
        major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

        j = 0
        for i in range(range1):
            if i > 0:
                conv_layer_name = 'conv2d_%d' %i
            else:
                conv_layer_name = 'conv2d'
                
            if j > 0:
                bn_layer_name = 'batch_normalization_%d' %j
            else:
                bn_layer_name = 'batch_normalization'
            
            conv_layer = model.get_layer(conv_layer_name)
            filters = conv_layer.filters
            k_size = conv_layer.kernel_size[0]
            in_dim = conv_layer.input_shape[-1]

            if i not in range2:
                # darknet weights: [beta, gamma, mean, variance]
                bn_weights = np.fromfile(wf, dtype=np.float32, count=4 * filters)
                # tf weights: [gamma, beta, mean, variance]
                bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
                bn_layer = model.get_layer(bn_layer_name)
                j += 1
            else:
                conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)

            # darknet shape (out_dim, in_dim, height, width)
            conv_shape = (filters, in_dim, k_size, k_size)
            conv_weights = np.fromfile(wf, dtype=np.float32, count=np.product(conv_shape))
            # tf shape (height, width, in_dim, out_dim)
            conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

            if i not in range2:
                conv_layer.set_weights([conv_weights])
                bn_layer.set_weights(bn_weights)
            else:
                conv_layer.set_weights([conv_weights, conv_bias])

        assert len(wf.read()) == 0, 'failed to read all data'


def image_resize(images, target_size, gt_boxes=None):
    if gt_boxes is None:
        images = tf.image.resize_with_pad(images, target_size[0], target_size[1], antialias=False).numpy()
        return images / 255.
    else:
        image_paded_list, gt_boxes_list = [], []
        for image in images:
            ih, iw    = target_size
            h,  w, _  = image.shape

            scale = min(iw/w, ih/h)
            nw, nh  = int(scale * w), int(scale * h)
            image_resized = cv2.resize(image, (nw, nh))

            image_paded = np.full(shape=[ih, iw, 3], fill_value=0.0)
            dw, dh = (iw - nw) // 2, (ih-nh) // 2
            image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
            image_paded = image_paded / 255.

            gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
            gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh

            image_paded_list.append(image_paded)
            gt_boxes_list.append(gt_boxes)

        return image_paded_list, gt_boxes_list


def draw_bbox(image, bboxes, class_names=YOLO_CLASSES, show_label=True, show_confidence = True, Text_colors=(255,255,0), rectangle_colors=''):   
    NUM_CLASS = read_class_names(class_names)
    num_classes = len(NUM_CLASS)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    #print("hsv_tuples", hsv_tuples)
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        score = bbox[4]
        class_ind = int(bbox[5])
        bbox_color = rectangle_colors if rectangle_colors != '' else colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 1000)
        if bbox_thick < 1: bbox_thick = 1
        fontScale = 0.75 * bbox_thick
        (x1, y1), (x2, y2) = (coor[0], coor[1]), (coor[2], coor[3])

        # put object rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color, bbox_thick*2)

        if show_label:
            # get text label
            score_str = " {:.2f}".format(score) if show_confidence else ""

            label = "{}".format(NUM_CLASS[class_ind]) + score_str

            # get text size
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                                  fontScale, thickness=bbox_thick)
            # put filled text rectangle
            cv2.rectangle(image, (x1, y1), (x1 + text_width, y1 - text_height - baseline), bbox_color, thickness=cv2.FILLED)

            # put text above rectangle
            cv2.putText(image, label, (x1, y1-4), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        fontScale, Text_colors, bbox_thick, lineType=cv2.LINE_AA)

    return image


def bboxes_iou(boxes1, boxes2):
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up       = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down    = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area    = inter_section[..., 0] * inter_section[..., 1]
    union_area    = boxes1_area + boxes2_area - inter_area
    ious          = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return ious

def ensembleboxes(bboxes, iou_threshold, sigma=0.3, skip_box_thr=0.0001, method=YOLO_METHOD_ENSEMBLEBOXES):
    boxes_list = np.array([bboxes[:,0:4].tolist()])
    scores_list = np.array([bboxes[:,4].tolist()])
    labels_list = np.array([bboxes[:,5].tolist()])

    if boxes_list.shape[-1] == 0:
        return []
    max_value = np.amax(boxes_list)
    boxes_list /= max_value

    boxes, scores, labels = None, None, None

    
    if method == 'nms':
        boxes, scores, labels = ensemble_boxes.nms(boxes_list, scores_list, labels_list, iou_thr=iou_threshold)
    elif method == 'soft_nms':
        boxes, scores, labels = ensemble_boxes.soft_nms(boxes_list, scores_list, labels_list, iou_thr=iou_threshold, sigma=sigma, thresh=skip_box_thr)
    elif method == 'nmw': # non_maximum_weighted
        boxes, scores, labels = ensemble_boxes.non_maximum_weighted(boxes_list, scores_list, labels_list, iou_thr=iou_threshold, skip_box_thr=skip_box_thr)
    elif method == 'wbf': # weighted_boxes_fusion
        boxes, scores, labels = ensemble_boxes.weighted_boxes_fusion(boxes_list, scores_list, labels_list, iou_thr=iou_threshold, skip_box_thr=skip_box_thr)
    else:
        assert False

    boxes *= max_value
    return np.concatenate((boxes, np.array([scores]).T, np.array([labels]).T), axis=1)


def postprocess_boxes(pred_bbox, original_image, input_size, score_threshold):
    valid_scale=[0, np.inf]
    pred_bbox = np.array(pred_bbox)

    pred_xywh = pred_bbox[:, 0:4]
    pred_conf = pred_bbox[:, 4]
    pred_prob = pred_bbox[:, 5:]

    # 1. (x, y, w, h) --> (xmin, ymin, xmax, ymax)
    pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)
    # 2. (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
    org_h, org_w = original_image.shape[:2]
    resize_ratio = min(input_size / org_w, input_size / org_h)

    dw = (input_size - resize_ratio * org_w) / 2
    dh = (input_size - resize_ratio * org_h) / 2

    pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
    pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

    # 3. clip some boxes those are out of range
    pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
    invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
    pred_coor[invalid_mask] = 0

    # 4. discard some invalid boxes
    bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
    scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

    # 5. discard boxes with low scores
    classes = np.argmax(pred_prob, axis=-1)
    scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
    score_mask = scores > score_threshold
    mask = np.logical_and(scale_mask, score_mask)
    coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]

    return np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)

# def pre_processing(images_list, input_size=YOLO_INPUT_SIZE):
#     images_data, original_images = [], []

#     for image_item in images_list:
#         original_image = cv2.imread(image_item) if isinstance(image_item, str) else image_item
#         original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
#         original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

#         image_data = image_resize(np.copy(original_image), [input_size, input_size])
#         image_data = image_data[np.newaxis, ...].astype(np.float32)

#         images_data.append(image_data)
#         original_images.append(original_image)

#     images_data = np.array([image_data.reshape(image_data.shape[1:]) for image_data in images_data])
#     original_images = np.array(original_images)

#     return images_data, original_images

def pre_processing(images_list, input_size=YOLO_INPUT_SIZE):
    
    original_images = []
    for image_item in images_list:
        original_image = cv2.imread(image_item) if isinstance(image_item, str) else image_item
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        original_images.append(original_image)
    original_images = np.array(original_images)

    images_data = image_resize(original_images, [input_size, input_size])

    return images_data, original_images


def predict(model, images_data):
    if YOLO_FRAMEWORK == "tf":
        return model.predict(images_data)
    elif YOLO_FRAMEWORK == "trt":
        # Não testado reecbendo multi imagens
        batched_input = tf.constant(images_data)
        result = model(batched_input)
        pred_bbox = []
        for key, value in result.items():
            value = value.numpy()
            pred_bbox.append(value)
        return pred_bbox
    return None


def post_processing(pred_bboxs, original_images, input_size=YOLO_INPUT_SIZE, score_threshold=0.3, iou_threshold=0.45, method=YOLO_METHOD_ENSEMBLEBOXES):
    
    number_of_images = len(original_images)
    number_of_anchors = len(pred_bboxs)

    pred_bbox_per_image = [None] * number_of_images

    for i in range(len(pred_bbox_per_image)):
        pred_bbox_per_image[i] = [None] * number_of_anchors

    for i1 in range(number_of_anchors):
        for i2 in range(number_of_images):
            pred_bbox_per_image[i2][i1]
            pred_bboxs[i1][i2]
            pred_bbox_per_image[i2][i1] = pred_bboxs[i1][i2]

    bboxes_list = []

    for pred_bbox, original_image in zip(pred_bbox_per_image, original_images):
        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = tf.concat(pred_bbox, axis=0)

        bboxes = postprocess_boxes(pred_bbox, original_image, input_size, score_threshold)
        bboxes = ensembleboxes(bboxes, iou_threshold, method=method)
        
        bboxes_list.append(bboxes)

    return bboxes_list


def show_image(image, title, wait=0):
    cv2.imshow(title, image)
    if wait == 0:
        cv2.waitKey(wait)
        cv2.destroyAllWindows()
    else:
        if cv2.waitKey(wait) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            return False
    return True

def save_image(image, image_filename_path):
    cv2.imwrite(image_filename_path, image)

def detect_image(model, images_list, output_path=None, input_size=YOLO_INPUT_SIZE, show=False, return_images=False, class_names=YOLO_CLASSES, score_threshold=0.3, iou_threshold=0.45, rectangle_colors='', method=YOLO_METHOD_ENSEMBLEBOXES):
    if not isinstance(images_list, list):
        images_list = [images_list]

    images_data, original_images = pre_processing(images_list=images_list, input_size=input_size)
    pred_bboxs = predict(model=model, images_data=images_data)
    bboxes_list = post_processing(pred_bboxs=pred_bboxs, original_images=original_images, input_size=input_size, score_threshold=score_threshold, iou_threshold=iou_threshold, method=method)

    if return_images or output_path is not None or show:
        images = []
        for i, (bboxes, image_path, original_image) in enumerate(zip(bboxes_list, images_list, original_images)):
            image = draw_bbox(original_image, bboxes, class_names=class_names, rectangle_colors=rectangle_colors)
            images.append(image)
            
            if output_path is not None:
                # O que acontece quando envia um imagem como objeto e não um path? Deve dar erro.
                image_filename = os.path.basename(image_path)
                image_filename_path = os.path.join(output_path, image_filename)
                save_image(image, image_filename_path)

            if show:
                image_filename = os.path.basename(image_path) if isinstance(image_path, str) else f'Image {i}'
                show_image(image, image_filename)

        if return_images:
            return bboxes_list, images

    return bboxes_list


def detect_video(model, filename_video_path, output_path, input_size=YOLO_INPUT_SIZE, show=False, class_names=YOLO_CLASSES, chunksize=1, score_threshold=0.3, iou_threshold=0.45, rectangle_colors=''):
    times, times_2 = [], []
    vid = cv2.VideoCapture(filename_video_path)

    if output_path is not None:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*'XVID')
        filename_video = os.path.basename(filename_video_path)
        filename_video = filename_video.split('.')
        filename_video[-1] = 'mp4'  # output_path must be .mp4
        filename_video = '.'.join(filename_video)
        filename_output_path = os.path.join(output_path, filename_video)
        out = cv2.VideoWriter(filename_output_path, codec, fps, (width, height))

    if chunksize is None:
        chunksize = 1
    while True:
        images_video = []
        for _ in range(chunksize):
            _, img = vid.read()
            if img is None:
                break
            images_video.append(img)
        if len(images_video) == 0:
            break

        return_images = output_path is not None or show
        ret_detect_image = detect_image(model, images_video, None, input_size=YOLO_INPUT_SIZE, show=False, return_images=return_images, class_names=class_names,
            score_threshold=score_threshold, iou_threshold=iou_threshold, rectangle_colors=rectangle_colors)
        bboxes_list, images = ret_detect_image if return_images else [ret_detect_image, None]
        
        if output_path is not None:
            [out.write(image) for image in images]

        if show:
            filename_video = os.path.basename(filename_video_path)
            for image in images:
                if not show:
                    break
                show = show_image(image, filename_video, wait=25)

    if show:
        cv2.destroyAllWindows()

