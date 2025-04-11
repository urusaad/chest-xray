import pandas as pd 
import numpy as np
import joblib 
import streamlit as st 
import cv2 as cv 
from PIL import Image 

# Load the model
model = joblib.load('Support_VM_model.pkl')

# Function to process image
def convert_image(image): 
    # Read the uploaded file into a NumPy array
    file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
    img = cv.imdecode(file_bytes, cv.IMREAD_GRAYSCALE)  # Convert to grayscale
    
    if img is None:
        st.error("Error: Unable to process image. Please upload a valid image file.")
        return None

    img_res = cv.resize(img, (80, 8))  # Resize to (80,80)
    img_array = img_res.flatten()  # Flatten the image
    img_df = pd.DataFrame([img_array])  # Convert to a single-row DataFrame
    return img_df

# Streamlit UI
st.title("Please Upload Your Image")
image = st.file_uploader("Upload image", type=['jpg', 'png'])

if image is not None:
    img_df = convert_image(image)

    if img_df is not None:
        pred = model.predict(img_df)  # Make prediction
        st.write("Prediction:", pred)
    else:
        st.warning("Please upload a valid image for prediction.")