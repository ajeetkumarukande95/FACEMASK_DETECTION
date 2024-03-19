import gradio as gr
import pickle
import numpy as np
from PIL import Image

# Load model from .pkl file
with open('model.pkl', 'rb') as f:
    model_pkl = pickle.load(f)

# Define a function to make predictions on a single image
def predict_single_image(image):
    # Resize and preprocess the image
    img = image.resize((128, 128))
    img = np.array(img) / 255.0  # Normalization
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Make prediction using the provided model
    prediction = model_pkl.predict(img)
    
    # Thresholding prediction
    threshold = 0.5
    prediction_class = (prediction > threshold).astype(int)

    # Interpret prediction
    if prediction_class == 1:
        return "With Mask"
    else:
        return "Without Mask"

# Create a Gradio interface
image = gr.inputs.Image(shape=(128, 128), label="Upload a photo of a person")
label = gr.outputs.Label(label="Prediction")

# Create Gradio interface with predict_single_image function
title = "Face Mask Detection"
description = "This app detects whether a person in the uploaded image is wearing a face mask or not. Created by Ajeetkumar Ukande."
author = "Ajeetkumar Ukande"
gr.Interface(predict_single_image, inputs=image, outputs=label, title=title, description=description, author=author, share=True, debug=True).launch()
