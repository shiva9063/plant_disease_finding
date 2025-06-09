# importing dependices

import tensorflow as tf
from tensorflow.keras.utils import save_img
from src.components.model_trainer import ModelTrainerConfig, ModelTrainer

import os
from src.logger import logging
import sys
from src.exception import CustomException
from dataclasses import dataclass

@dataclass  #  the __init__ method is automatically generated
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train')
    test_data_path: str = os.path.join('artifacts', 'test')
    val_data_path: str = os.path.join('artifacts', 'validate')
    raw_data_path: str = os.path.join('artifacts', 'raw')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Reading the data...')
        try:
            IMAGE_SIZE = 225
            BATCH_SIZE = 8  # the total images divide into 8 batches

            dataset = tf.keras.preprocessing.image_dataset_from_directory(
                r'C:\Users\n shiva kumar\OneDrive\Desktop\image_classification\notebook\data',
                image_size=(IMAGE_SIZE, IMAGE_SIZE),
                batch_size=BATCH_SIZE
            )
        
        
            logging.info('Dataset loaded successfully.')

            # Dataset partitioning
            def get_dataset_partitions(ds, train_split=0.8, test_split=0.1, val_split=0.1, shuffle=True):
                ds_size = len(ds)
                if shuffle:
                    ds = ds.shuffle(1000, seed=12)
                train_size = int(ds_size * train_split)
                val_size = int(ds_size * val_split)

                train_ds = ds.take(train_size)
                test_ds = ds.skip(train_size).take(val_size)
                val_ds = ds.skip(train_size).skip(val_size)
                return train_ds, test_ds, val_ds

            train_ds, test_ds, val_ds = get_dataset_partitions(dataset)
            class_names = dataset.class_names

            # Only save dataset if path doesn't already exist
            def save_dataset_if_not_exists(dataset, dataset_path):
                if os.path.exists(dataset_path):
                    logging.info(f"Skipped saving: {dataset_path} already exists.")
                    return

                for batch_idx, (images, labels) in enumerate(dataset):
                    for i in range(len(images)):
                        img = images[i].numpy()
                        label = labels[i].numpy()
                        class_name = class_names[label]

                        class_dir = os.path.join(dataset_path, class_name)
                        os.makedirs(class_dir, exist_ok=True)

                        filename = f"{class_name}_{batch_idx}_{i}.jpg"
                        save_path = os.path.join(class_dir, filename)
                        if not os.path.exists(save_path):
                            save_img(save_path, img)
                logging.info(f"Saved dataset to {dataset_path}")

            # Save datasets (if not already saved)
            save_dataset_if_not_exists(train_ds, self.ingestion_config.train_data_path)
            save_dataset_if_not_exists(test_ds, self.ingestion_config.test_data_path)
            save_dataset_if_not_exists(val_ds, self.ingestion_config.val_data_path)
            save_dataset_if_not_exists(dataset, self.ingestion_config.raw_data_path)

            logging.info('All datasets processed.')
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                self.ingestion_config.val_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)

# Run block
if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data, val_data = obj.initiate_data_ingestion()

    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_data, test_data, val_data))
