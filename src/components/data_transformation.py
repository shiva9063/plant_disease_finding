import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import models,layers
from tensorflow import keras
from tensorflow.keras.utils import save_img


import os
from src.logger import logging
import sys
from src.exception import CustomException
from dataclasses import dataclass

class DataTransformationConfig:
    preprocess_object_file_path=os.path.join('artifacts','preprocessor.keras')
    agumentation_object_file_path=os.path.join('artifacts','agumentation.keras')
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    def get_data_transformer_obj(self):
        try:
            IMG_SIZE=224
            resize_and_rescale=tf.keras.Sequential([
                layers.experimental.preprocessing.Resizing(IMG_SIZE,IMG_SIZE),
                layers.experimental.preprocessing.Rescaling(1.0/224)

            ])
            logging.info('resizing completed')
            data_agumentation=tf.keras.Sequential([
                layers.experimental.preprocessing.RandomFlip('horizantal_and_vertical'),
                layers.experimental.preprocessing.RandomRotation(0.2)
            ])
            logging.info('data agumentation is completed')
            return resize_and_rescale,data_agumentation
        except Exception as e:
            raise CustomException(e,sys)
    def initiate_data_transformation(self,train_path,test_path)

