import tensorflow as tf 
import numpy as np
import zipfile
import os 
import streamlit as st
from PIL import Image
import pathlib

# Load the model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('best_model.keras')
    return model
model = load_model()

# Extracting the elements contained in the zipfile
dir_name = 'Brain tumour classification kaggle.zip'
zipfile = zipfile.ZipFile(dir_name, 'r')
zipfile.extractall()
zipfile.close()

# Getting the class names
directory_path = pathlib.Path('Brain tumour classification kaggle')
class_names = list(sorted([item.name for item in directory_path.glob('Training/*')]))


# Streamlit app
st.title("Brain Tumor Classification")
st.write("Upload an image of a brain MRI to classify it as one of the following categories:")
st.write(class_names)
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    st.write("")
    st.write("Classifying...")
    
    img = image.resize((224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    

    prediction = model.predict(img)
    pred_class_index = np.argmax(prediction[0])
    pred_confidence = prediction[0][pred_class_index]
    
    st.write(f"Predicted Class: {class_names[pred_class_index]}")
    st.write(f"Confidence: {pred_confidence:.2f}")


