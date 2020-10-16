import tensorflow as tf
import numpy as np
import os
from functools import reduce
from tensorflow.keras.layers import Conv2D,MaxPooling2D,UpSampling2D,LeakyReLU,Concatenate,Add,ZeroPadding2D,BatchNormalization,Input
from keras.models import Model
from tensorflow.keras.regularizers import L2
import tensorflow.keras.backend as K



def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError("Composition of empty sequence not supported.")


def conv2d(no_filter,filter_size, **kwargs):
    """ Convolutional layer of the darknet"""
    if kwargs.get("stride") == (2,2):
        y = "valid"
        x = (2,2)
    else:
        y = "same"
        x = (1,1)
    conv = Conv2D(no_filter, filter_size, strides = x, padding= y ,use_bias=False , kernel_regularizer=L2(0.0005 ))
    conv = BatchNormalization() (conv)
    conv = LeakyReLU(alpha = 0.1)(conv)
    return conv


def residual_block(x,no_filter,no_times):
    """residual block which is used in the darknet"""
    x = ZeroPadding2D(((1,0),(1,0)))(x)
    x = conv2d(no_filter,(3 , 3),stride=(2,2))(x)
    for a in range(no_times):
        y = compose(conv2d(no_filter//2 , (1,1)) , conv2d(no_filter , (3 ,3))) (x)

        x = Add()([x , y])
    return x


def darknet53(x):
    """Darknet Body having 52 Conv2d layers"""
    x = conv2d(32,(3,3))(x)
    x = residual_block(x , 64 , 1)
    x = residual_block(x , 128 , 2)
    x = residual_block(x , 256 , 8)
    x2 = x
    x = residual_block(x , 512 , 8)
    x1 = x
    x = residual_block(x , 1024 , 4)
    return x , x1 , x2

def make_last_layer(x , no_filter, out_filter):
    """this is last layer consisting 5 DBL followed by Convolutional layer"""
    x = compose(conv2d(no_filter,(1,1)),
                conv2d(no_filter*2,(3,3)),
                conv2d(no_filter,(1,1)),
                conv2d(no_filter*2,(3,3)),
                conv2d(no_filter,(1,1)))(x)
    y = compose(conv2d(no_filter*2,(3,3)),
                Conv2D(out_filter,(1,1),strides=(1,1),padding='same',use_bias=False,kernel_regularizer=L2(0.0005))
                )(x)

    return x, y

def yolov3(input,no_anchors,no_classes):
    """YOLOV3 CNN taking arguments Input,no_anchors and no_classes"""
    x , x1 , x2 = darknet53(input)

    x , y1 = make_last_layer(x , 512 , no_anchors*(no_classes+5))

    x = compose(conv2d(256,(1,1)),UpSampling2D(2))(x)
    x = Concatenate()([x , x1])
    x , y2 = make_last_layer(x , 256 , no_anchors*(no_classes+5))

    x = compose(conv2d(256,(1,1)),UpSampling2D(2))(x)
    x = Concatenate()([x , x2])
    x , y3 = make_last_layer(x , 256 , no_anchors*(no_classes+5))

    return Model(input ,[y1 , y2 , y3])

def decode(yolo_out, anchors, num_classes, input_shape, calc_loss=False):
    """Takes the YOLO out and predicts the bounding boxes"""
    anchor_tensor = tf.keras.backend.reshape(tf.keras.backend.constant(anchors),
                                                [1 , 1 , 1 , 3 , 2])

    yolo_shape = K.shape(yolo_out)[1:3] #height and width
    

    yolo_out = K.reshape(yolo_out,(-1 , yolo_shape[0] , yolo_shape[1] , 3 , 5+num_classes)) #here 3 is the no of anchors
       
    t_xy = yolo_out[: , : , : , : , 0:2] #x and y position from the center

    t_wh = yolo_out[: , : , : , : , 2:4] #box width and height

    confidence = yolo_out[: , : , : , : , 4:5] # confidence if the box

    class_prob = yolo_out[: , : , : , : , 5: ] #class probablities

    grid_y = K.tile(
        K.reshape(K.arange(0, stop=yolo_shape[0]), [-1, 1, 1, 1]),
        [1, yolo_shape[1], 1, 1],
    )
    grid_x = K.tile(
        K.reshape(K.arange(0, stop=yolo_shape[1]), [1, -1, 1, 1]),
        [yolo_shape[0], 1, 1, 1],
    )
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(yolo_out))

    #Calculate bounding box xy
    b_xy = (K.sigmoid(t_xy) + grid)/K.cast(yolo_shape[::-1], K.dtype(yolo_out))

    b_wh = (K.exp(t_wh)*anchor_tensor)/K.cast(input_shape[::-1], K.dtype(yolo_out))

    b_c = K.sigmoid(confidence)

    b_prob = K.sigmoid(class_prob)

    if calc_loss == True:
        return grid, yolo_out, b_xy, b_wh
    return b_xy, b_wh, b_c, b_prob


def boundbox_correction(b_xy , b_wh , input_shape , image_shape):
    """Get corrected boxes"""
    box_yx = b_xy[..., ::-1]
    box_hw = b_wh[..., ::-1]
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))
    new_shape = K.round(image_shape * K.min(input_shape / image_shape))
    offset = (input_shape - new_shape) / 2.0 / input_shape
    scale = input_shape / new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.0)
    box_maxes = box_yx + (box_hw / 2.0)
    boxes = K.concatenate(
        [
            box_mins[..., 0:1],  # y_min
            box_mins[..., 1:2],  # x_min
            box_maxes[..., 0:1],  # y_max
            box_maxes[..., 1:2],  # x_max
        ]
    )

    # Scale boxes back to original image shape.
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes

def yolo_boxes_and_scores(yolo_out, anchors, num_classes, input_shape, image_shape):
    """Process Conv layer output"""
    box_xy, box_wh, box_confidence, box_class_probs = decode(
        yolo_out, anchors, num_classes, input_shape
    )
    boxes = boundbox_correction(box_xy, box_wh, input_shape, image_shape)
    boxes = K.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = K.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores


def evaluate_yolo(yolo_out,anchors,num_classes,image_shape,max_boxes=20,score_threshold=0.6,iou_threshold=0.5):
    """Evaluate YOLO model on given input and return filtered boxes."""
    num_layers = len(yolo_out)
    anchor_mask = (
        [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]
    )  # default setting
    input_shape = K.shape(yolo_out[0])[1:3] * 32
    boxes = []
    box_scores = []
    for l in range(num_layers):
        _boxes, _box_scores = yolo_boxes_and_scores(
            yolo_out[l],
            anchors[anchor_mask[l]],
            num_classes,
            input_shape,
            image_shape,
        )
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    boxes = K.concatenate(boxes, axis=0)
    box_scores = K.concatenate(box_scores, axis=0)

    mask = box_scores >= score_threshold
    max_boxes_tensor = K.constant(max_boxes, dtype="int32")
    boxes_ = []
    scores_ = []
    classes_ = []

    for c in range(num_classes):
        # TODO: use keras backend instead of tf.
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold
        )
        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, "int32") * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = K.concatenate(boxes_, axis=0)
    scores_ = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)

    return boxes_, scores_, classes_

def box_iou(b1, b2):
    """Return iou tensor

    Parameters
    ----------
    b1: tensor, shape=(i1,...,iN, 4), xywh
    b2: tensor, shape=(j, 4), xywh

    Returns
    -------
    iou: tensor, shape=(i1,...,iN, j)

    """

    # Expand dim to apply broadcasting.
    b1 = K.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh / 2.0
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    b2 = K.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh / 2.0
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.0)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou


def yolo_loss(args, anchors, num_classes, ignore_thresh=0.5, print_loss=False):
    """Return yolo_loss tensor

    Parameters
    ----------
    yolo_outputs: list of tensor, the output of yolo_body or tiny_yolo_body
    y_true: list of array, the output of preprocess_true_boxes
    anchors: array, shape=(N, 2), wh
    num_classes: integer
    ignore_thresh: float, the iou threshold whether to ignore object confidence loss

    Returns
    -------
    loss: tensor, shape=(1,)

    """
    num_layers = len(anchors) // 3  # default setting
    yolo_outputs = args[:num_layers]
    y_true = args[num_layers:]
    anchor_mask = (
        [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]
    )
    input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))
    grid_shapes = [
        K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(y_true[0]))
        for l in range(num_layers)
    ]
    loss = 0
    m = K.shape(yolo_outputs[0])[0]  # batch size, tensor
    mf = K.cast(m, K.dtype(yolo_outputs[0]))

    for l in range(num_layers):
        object_mask = y_true[l][..., 4:5]
        true_class_probs = y_true[l][..., 5:]

        grid, raw_pred, pred_xy, pred_wh = decode(
            yolo_outputs[l],
            anchors[anchor_mask[l]],
            num_classes,
            input_shape,
            calc_loss=True,
        )
        pred_box = K.concatenate([pred_xy, pred_wh])

        # Darknet raw box to calculate loss.
        raw_true_xy = y_true[l][..., :2] * grid_shapes[l][::-1] - grid
        raw_true_wh = K.log(
            y_true[l][..., 2:4] / anchors[anchor_mask[l]] * input_shape[::-1]
        )
        raw_true_wh = K.switch(
            object_mask, raw_true_wh, K.zeros_like(raw_true_wh)
        )  # avoid log(0)=-inf
        box_loss_scale = 2 - y_true[l][..., 2:3] * y_true[l][..., 3:4]

        # Find ignore mask, iterate over each of batch.
        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = K.cast(object_mask, "bool")

        def loop_body(b, ignore_mask):
            true_box = tf.boolean_mask(
                y_true[l][b, ..., 0:4], object_mask_bool[b, ..., 0]
            )
            iou = box_iou(pred_box[b], true_box)
            best_iou = K.max(iou, axis=-1)
            ignore_mask = ignore_mask.write(
                b, K.cast(best_iou < ignore_thresh, K.dtype(true_box))
            )
            return b + 1, ignore_mask

        _, ignore_mask = tf.while_loop(
            lambda b, *args: b < m, loop_body, [0, ignore_mask]
        )
        ignore_mask = ignore_mask.stack()
        ignore_mask = K.expand_dims(ignore_mask, -1)

        # K.binary_crossentropy is helpful to avoid exp overflow.
        xy_loss = (
            object_mask
            * box_loss_scale
            * K.binary_crossentropy(raw_true_xy, raw_pred[..., 0:2], from_logits=True)
        )
        wh_loss = (
            object_mask
            * box_loss_scale
            * 0.5
            * K.square(raw_true_wh - raw_pred[..., 2:4])
        )
        confidence_loss = (
            object_mask
            * K.binary_crossentropy(object_mask, raw_pred[..., 4:5], from_logits=True)
            + (1 - object_mask)
            * K.binary_crossentropy(object_mask, raw_pred[..., 4:5], from_logits=True)
            * ignore_mask
        )
        class_loss = object_mask * K.binary_crossentropy(
            true_class_probs, raw_pred[..., 5:], from_logits=True
        )

        xy_loss = K.sum(xy_loss) / mf
        wh_loss = K.sum(wh_loss) / mf
        confidence_loss = K.sum(confidence_loss) / mf
        class_loss = K.sum(class_loss) / mf
        loss += xy_loss + wh_loss + confidence_loss + class_loss
        if print_loss:
            tf.print('loss',loss,'xy_loss',xy_loss)
            tf.print('wh_loss',wh_loss,'confidence_loss',confidence_loss)
            tf.print('class_loss',class_loss,K.sum(ignore_mask))
                
    return loss