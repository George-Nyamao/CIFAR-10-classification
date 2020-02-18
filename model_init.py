import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

def init(model_dir, with_gpu = True):
    if with_gpu:
        gpus = tf.config.exxperimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            except RuntimeError as e:
                print(e)

def preprocess_img(filename):
    mean = -1.436480564128336e-17
    std = 0.9999999000000056
    img = load_img(filename, target_size=(32, 32))
    img = img_to_array(img)
    img = img.reshape(1, 32, 32, 3)
    img = img.astype('float32')
    img = (img-mean)/(std+1e-7)
    return img

def my_label(inference_result):
    label_index = inference_result[0]

    labels = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    mapping = labels[label_index]
    return mapping