import os
from PIL import Image
from cv2 import cv2
import tensorflow as tf
import numpy as np
import re
import matplotlib.pyplot as plt

def get_random_data(annotation_lines, input_shape):
    """this function is to get the labelled box and the image data"""
    
    tmp_split = re.split("( \d)", annotation_lines, maxsplit=1)
    if len(tmp_split) > 2:
        line = tmp_split[0], tmp_split[1] + tmp_split[2]
    else:
        line = tmp_split
    # line[0] contains the filename
    
    image = Image.open(line[0])
    # The rest of the line includes bounding boxes
    line = line[1].split(" ")
    
    box = np.array([np.array(list(map(int, box.split(",")))) for box in line[1:]])
    max_box = 20
    h , w = input_shape

    iw , ih  = image.size

    scale = min(w/iw , h/iw)
    nw ,nh= int(scale *iw) , int(scale * ih)
        
    dx = (w -nw) // 2
    dy = (h - nh) // 2
    image_data = 0
    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new("RGB", (w, h), (128, 128, 128))
    new_image.paste(image, (dx, dy))
    image_data = np.array(new_image) / 255.0

    box_data = np.zeros((max_box ,5))

    if len(box) > 0:
        np.random.shuffle(box)
        if len(box) > max_box:
            box = box[:max_box]
        box[:, [0, 2]] = box[:, [0, 2]] * scale + dx
        box[:, [1, 3]] = box[:, [1, 3]] * scale + dy
        box_data[: len(box)] = box

    return image_data, box_data

def preprocess_true_boxes(true_boxes,input_shape,anchors,num_classes):
    """this function is to process the labelled box data into the y_true format"""
        
    assert (
        true_boxes[..., 4] < num_classes
    ).all(), "class id must be less than num_classes"
    num_layers = len(anchors) // 3  # default setting
    anchor_mask = (
        [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]
    )

    true_boxes = np.array(true_boxes, dtype="float32")
    input_shape = np.array(input_shape, dtype="int32")
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]

    m = true_boxes.shape[0]
    grid_shapes = [input_shape // {0: 32, 1: 16, 2: 8}[l] for l in range(num_layers)]
    y_true = [
        np.zeros(
            (
                m,
                grid_shapes[l][0],
                grid_shapes[l][1],
                len(anchor_mask[l]),
                5 + num_classes,
            ),
            dtype="float32",
        )
        for l in range(num_layers)
    ]

    # Expand dim to apply broadcasting.
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.0
    anchor_mins = -anchor_maxes
    valid_mask = boxes_wh[..., 0] > 0

    for b in range(m):
        # Discard zero rows.
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh) == 0:
            continue
        # Expand dim to apply broadcasting.
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.0
        box_mins = -box_maxes

        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.0)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        # Find best anchor for each true box
        best_anchor = np.argmax(iou, axis=-1)

        for t, n in enumerate(best_anchor):
            for l in range(num_layers):
                if n in anchor_mask[l]:
                    i = np.floor(true_boxes[b, t, 0] * grid_shapes[l][1]).astype(
                        "int32"
                    )
                    j = np.floor(true_boxes[b, t, 1] * grid_shapes[l][0]).astype(
                        "int32"
                    )
                    k = anchor_mask[l].index(n)
                    c = true_boxes[b, t, 4].astype("int32")
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]
                    y_true[l][b, j, i, k, 4] = 1
                    y_true[l][b, j, i, k, 5 + c] = 1

    return y_true   

    
    