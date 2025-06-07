from flask import Flask, request, render_template
from PIL import Image
import tensorflow as tf
import numpy as np
import os
from src.pipeline.predict_pipeline import PredictPipeline

application = Flask(__name__)
app = application

# Load class names once from directory
IMAGE_SIZE = 255
BATCH_SIZE = 8
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    r'C:\Users\n shiva kumar\OneDrive\Desktop\image_classification\notebook\data',
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE
)
class_names = ['Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy']

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        return render_template('home.html')
            # Load image and preprocess
    else:
        image_file = request.files.get('file')
        if image_file:
            try:
                # Load image and preprocess
                img = Image.open(image_file).convert("RGB")
                img = img.resize((225, 225))
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)  # (1, 225, 225, 3)

                # DO NOT normalize again if model has Rescaling layer
                # img_array = img_array / 255.0  <-- comment or remove this

                # Load model
                model = tf.keras.models.load_model('artifacts/model.keras')

                # Predict
                prediction = model.predict(img_array)
                predicted_class = class_names[np.argmax(prediction)]

                return render_template('home.html', result=f"Predicted Class: {predicted_class}")
            except Exception as e:
                return render_template('home.html', result=f"Error: {str(e)}")


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
