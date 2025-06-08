🍎 Apple Leaf Disease Prediction using Deep Learning
This project implements an end-to-end deep learning pipeline to classify apple leaf diseases from images using Convolutional Neural Networks (CNNs). It helps farmers or agricultural experts diagnose leaf diseases early and take preventive action.

📌 Problem Statement
Apple plants are susceptible to various diseases, which often manifest visually on leaves. Manual identification is time-consuming and error-prone. This project uses deep learning to automatically classify apple leaf images into eight categories, including healthy and diseased classes.

📁 Dataset
Contains 4 classes of apple leaf images.

Classes include:

Apple Scab

Black Rot

Cedar Apple Rust

Healthy

and 4 more variants.

Data is split into:

Raw data

Train set

Test set

Dataset was curated and preprocessed before training.

🧠 Model
Framework: TensorFlow & Keras

Model: Custom CNN

Key Layers:

Convolutional layers

MaxPooling

Dropout for regularization

Fully connected dense layers

Optimizer: Adam

Loss: SparseCategoricalCrossentropy

Evaluation Metric: Accuracy

🔄 Data Pipeline
Image resizing: 225x225

Rescaling: [0, 1] normalization

Data Augmentation:

Horizontal and vertical flip

Random rotation

📊 Results
Achieved training and validation accuracy up to ~90% (depends on dataset and epochs)

Confusion matrix and classification report were generated for model evaluation

🖼️ Web App (Optional)
A simple Flask-based web interface allows users to upload an image of an apple leaf and get instant predictions with disease name.

📌 How to Run
🔧 Requirements
Python 3.10+

TensorFlow

NumPy, Matplotlib, Pandas

Flask (for web app)

