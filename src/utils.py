import numpy as np
import dill
import os
from src.exception import CustomException
from src.logger import logging
import sys
import tensorflow as tf
from sklearn.metrics import accuracy_score



def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    except:
        pass
def evaluate_models(train_path,val_path,test_path,BATCH_SIZE,model,EPOCH):
    try:
        
        def get_data(path,IMAGE_SIZE=225):
            data_set=tf.keras.preprocessing.image_dataset_from_directory(
               path,
               image_size=(IMAGE_SIZE, IMAGE_SIZE),
               batch_size=BATCH_SIZE
             )
            return data_set
        train_data_set=get_data(train_path)
        test_data_set=get_data(test_path)
        val_data_set=get_data(val_path)
        #model training
        history=model.fit(
        train_data_set,
        epochs=EPOCH,
        batch_size=BATCH_SIZE,
        verbose=1,
        validation_data=val_data_set
        )
        #history of data
        train_history=history.history
        EPOCH=1
        train_pred=train_history['accuracy'][EPOCH]
        val_pred=train_history['val_accuracy'][EPOCH]
        predict_list=[]
        actual_list=[]
        
        for image_batch, label_batch in test_data_set.take(38):
            pred=model.predict(image_batch)
            predict_list.extend([int(np.argmax(i)) for i in pred])
            actual_list.extend(label_batch)
        test_score=accuracy_score(predict_list,actual_list)
        return model,test_score
    except Exception as e:
        raise CustomException(e,sys)

def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)
