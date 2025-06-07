import os
import sys
import tensorflow as tf
from src.exception import CustomException
from src.utils import load_object
class PredictPipeline:
    def __init__(self):
        model_path = os.path.join("artifacts", "model.keras")  # or .pkl depending on your model format
        self.model = load_object(file_path=model_path)  # Your existing load_object function
        
    def predict(self,image_array):
        try:
            preds = self.model.predict(image_array)
            return preds
        except Exception as e:
            raise CustomException(e, sys)
