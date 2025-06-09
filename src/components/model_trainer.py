import numpy as np
import tensorflow as tf
from tensorflow.keras import models,layers
from tensorflow.keras.utils import save_img
from tensorflow.keras.models import Sequential
import keras
from src.utils import save_object,evaluate_models
import os
from src.logger import logging
import sys
from src.exception import CustomException
from dataclasses import dataclass
from sklearn.metrics import accuracy_score

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.keras')
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    def initiate_model_trainer(self,train_path,test_path,val_path):
        try:
            IMAGE_SIZE=225
            BATCH_SIZE=8
            EPOCH=10
            n_classes=8
            resize_and_rescale = Sequential([
            layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),
            layers.Rescaling(1.0 / 255)  # Normalizes pixel values to [0, 1]
            ])
            data_agumentation=tf.keras.Sequential([
            layers.RandomFlip('horizantal_and_vertical'),
            layers.RandomRotation(0.2)
            ])
            CHANNEL=3   #RGB
            input_shape=(BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,CHANNEL)
            model=models.Sequential([
                resize_and_rescale,
                data_agumentation,
                layers.Conv2D(32,(3,3),activation='relu',input_shape=input_shape),
                layers.MaxPooling2D((2,2)),

                layers.Conv2D(64,kernel_size=(3,3),activation='relu'),
                layers.MaxPooling2D((2,2)),
                layers.Conv2D(64,kernel_size=(3,3),activation='relu'),
                layers.MaxPooling2D((2,2)),

                layers.Conv2D(64,(3,3),activation='relu'),
                layers.MaxPooling2D((2,2)),
                layers.Conv2D(64,(3,3),activation='relu'),
                layers.MaxPooling2D((2,2)),
                layers.Conv2D(64,(3,3),activation='relu'),
                layers.MaxPooling2D((2,2)),

                layers.Flatten(),

                layers.Dense(64,activation='relu'),
                layers.Dense(n_classes,activation='softmax'),
            ]
            )
            model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=['accuracy']
               )
            best_model,best_model_score=evaluate_models(train_path,val_path,test_path,BATCH_SIZE,model=model,EPOCH=EPOCH)
            if best_model_score<0.6:
                raise CustomException('No Best Model Found')
            logging.info('we found a best model')
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predict_list=[]
            actual_list=[]
            test_data_set=tf.keras.preprocessing.image_dataset_from_directory(
               test_path,
               image_size=(IMAGE_SIZE, IMAGE_SIZE),
               batch_size=BATCH_SIZE
             )
            for image_batch, label_batch in test_data_set:
                pred=model.predict(image_batch)
                predict_list.extend([int(np.argmax(i)) for i in pred])
                actual_list.extend(label_batch)
            test_score=accuracy_score(predict_list,actual_list)
            return test_score
        except Exception as e:
            raise CustomException(e,sys)
        

