# Import libraries and dependencies
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

# If using GPUs, limit resources as needed
def init(model_dir, with_gpu = True):
    if with_gpu:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
          try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
              tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
          except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

# Image normalization
def preprocess_img(filename):
    mean = -1.436480564128336e-17
    std = 0.9999999000000056
    img = load_img(filename, target_size=(32, 32))
    img = img_to_array(img)
    img = img.reshape(1, 32, 32, 3)
    img = img.astype('float32')
    img = (img-mean)/(std+1e-7)
    return img

# Image categorization
def my_label(predict_result):
    label_index = predict_result[0]

    labels = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    label = labels[label_index]
    return label