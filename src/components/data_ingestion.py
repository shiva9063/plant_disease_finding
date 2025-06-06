import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import models,layers
from tensorflow.keras.utils import save_img


import os
from src.logger import logging
import sys
from src.exception import CustomException
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts','train')
    test_data_path:str=os.path.join('artifacts','test')
    raw_data_path:str=os.path.join('artifacts','raw')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    def initiate_data_ingestion(self):
        logging.info('reading the data')
        try:
            IMAGE_SIZE=225
            BATCH_SIZE=8
            # Move from 'src/components' up to 'notebook/data/plant_data'
            dataset = tf.keras.preprocessing.image_dataset_from_directory(
               'C:\\Users\\n shiva kumar\\OneDrive\\Desktop\\image_classification\\notebook\\data',
               image_size=(IMAGE_SIZE, IMAGE_SIZE),
               batch_size=BATCH_SIZE
             )

            logging.info('data as exported')
            # Save directory
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)

            # data partitioning
            def get_dataset_partions_df(ds,train_split=0.8,test_split=0.2,shuffle=True,shuffle_size=1000):
                ds_size=len(ds)
                train_size=int(ds_size*train_split)

                train_ds=ds.take(train_size)
                test_ds=ds.skip(train_size)
                return train_ds,test_ds
            train_ds,test_ds=get_dataset_partions_df(dataset)
            logging.info('data set splitted')
            class_names = dataset.class_names  # Assume same class names in both

            # Save function
            def save_dataset(dataset, dataset_path):
                for batch_idx, (images, labels) in enumerate(dataset):
                    for i in range(len(images)):
                        img = images[i].numpy()
                        label = labels[i].numpy()
                        class_name = class_names[label]

                        class_dir = os.path.join(dataset_path, class_name)
                        os.makedirs(class_dir, exist_ok=True)

                        filename = f"{class_name}_{batch_idx}_{i}.jpg"
                        save_path = os.path.join(class_dir, filename)
                        save_img(save_path, img)
            # Save datasets to train and test paths
            logging.info('train and test datasaved')
            save_dataset(train_ds, self.ingestion_config.train_data_path)
            save_dataset(test_ds, self.ingestion_config.test_data_path)
            save_dataset(dataset,self.ingestion_config.raw_data_path)
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
if __name__=="__main__":
    obj=DataIngestion()
    obj.initiate_data_ingestion()