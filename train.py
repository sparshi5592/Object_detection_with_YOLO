import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Lambda
from tensorflow.keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping,TensorBoard
from tensorflow.keras.optimizers import Adam
import os

from yolo import yolov3,yolo_loss
from get_image_data import get_random_data, preprocess_true_boxes

def get_anchors(anchor_path):
    """red the anchor.txt file"""
    with open(anchor_path) as a:
        anchors = a.readline()
        
    anchors = [float(x) for x in anchors.split(",")]
    return np.array(anchors).reshape(-1,2)


def get_classes(class_path):
    """to read the classes file"""
    with open(class_path) as d:
        classes = d.readlines()
    classes = [c.strip() for c in classes]
    return classes

def get_annotations(annotation_path):
    """ to open and read the annotation file"""
    with open(annotation_path) as f:
        annotation_lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(annotation_lines)
    np.random.seed(None)
    return annotation_lines

def create_model(input_shape,no_classes,anchors,load_pretrained_weight = True,freeze_body = 2 , weight_path = 'CWD + .txt'): #please add weight path next
    """this function is to create a yolo Model"""
    K.clear_session()
    image_shape = Input(shape=(None,None,3))
    h ,w = input_shape
    no_anchors = len(anchors)//3
    
    yolo_body = yolov3(image_shape , no_anchors , no_classes)
    
    y_true = [
        Input(
            shape=(
                h // {0: 32, 1: 16, 2: 8}[l],
                w // {0: 32, 1: 16, 2: 8}[l],
                no_anchors,
                no_classes + 5,
            )
        )
        for l in range(3)
    ]
    
    if load_pretrained_weight:
        yolo_body.load_weights(weight_path , by_name = True, skip_mismatch = True)
        print("Load weights {}.".format(weight_path))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(yolo_body.layers) - 3)[freeze_body - 1]
            for i in range(num):
                yolo_body.layers[i].trainable = False
            print(
                "Freeze the first {} layers of total {} layers.".format(
                    num, len(yolo_body.layers)
                )
            )

    loss = Lambda(yolo_loss,output_shape=(1 , ),
                    name = "yolo_loss",
                    arguments = {"anchors" :anchors ,
                                "num_classes" : no_classes,
                                "ignore_thresh":0.5},)([*yolo_body.output, *y_true])
    
    model = Model([yolo_body.input , *y_true],loss )
    return model

def get_data(annotation_lines,batch_size,input_shape,anchors,no_classes):
    """ This function is to get the image and the box data for training"""
    while True:
        image_data = []
        box_data = []
        for x in range(len(annotation_lines)-1):
            image, box = get_random_data(annotation_lines[x+1], input_shape)
            box_data.append(box)    
            image_data.append(image)

        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data,input_shape,anchors,no_classes)

        yield [image_data, *y_true], np.zeros(batch_size)
    

    

def train(testing = True):
    CWD = os.getcwd()
    anchor_path =CWD + '/model_data/yolo_anchors.txt'
      
    classes_path = CWD + '/model_data/data_classes.txt'
    annotation_path = CWD + '/model_data/data_train.txt'
    weight_path = CWD # + path of the pretrained weight
    log_dir = CWD + '/data_log/'
    anchors = get_anchors(anchor_path)
    classes = get_classes(classes_path)
    annotation_lines = get_annotations(annotation_path)

    input_shape = (416,416)
    no_classes = len(classes)
    model = create_model(input_shape,no_classes,anchors,load_pretrained_weight = False)
    
    no_annotations = len(annotation_lines)
    
    logging = TensorBoard(log_dir=log_dir)
    
    checkpoint = ModelCheckpoint(
        log_dir + "checkpoint.h5",
        monitor="val_loss",
        save_weights_only=True,
        save_best_only=True,
        period=5,
    )
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=3, verbose=1)

    early_stopping = EarlyStopping(
        monitor="val_loss", min_delta=0, patience=10, verbose=1
    )
    #this is check if the training is going in proper way and to adjust the epochs with first 5 data set
    if testing:
        
        model.compile(optimizer= Adam(lr = 0.001),
                        loss= {"yolo_loss":lambda y_true, y_pred:y_pred})

        batch_size = 32
        
        model.fit_generator(get_data(annotation_lines,batch_size,input_shape,anchors,no_classes),
                            steps_per_epoch=1,
                            callbacks=[logging,checkpoint],
                            epochs= 50,initial_epoch= 0,
                            validation_data= get_data(annotation_lines[:10],batch_size,input_shape,anchors,no_classes),
                            validation_steps=1)
        
        model.save_weights(log_dir + "trail_1_training.h5")

    if not testing:
        num_layers = len(model.layers)

        for x in range(num_layers):
            model.layers[x].trainable = True
        
    
        model.compile(optimizer= Adam(lr = 0.001),
                        loss= {"yolo_loss":lambda y_true, y_pred:y_pred})

        print("Training started with unfreezing and with {} samples and with the batch size of {} ." .format(no_annotations,batch_size))
        batch_size = 16
        
        model.fit_generator(get_data(annotation_lines,batch_size,input_shape,anchors,no_classes),
                            steps_per_epoch=1,
                            callbacks=[logging,checkpoint,reduce_lr,early_stopping],
                            epochs= 100,initial_epoch= 50,
                            validation_data= get_data(annotation_lines[:10],batch_size,input_shape,anchors,no_classes),
                            validation_steps=1)
        
        model.save_weights(log_dir + "final_trained_weights.h5")

    
    

if __name__ == "__main__":
    train()

    