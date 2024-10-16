import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained model
model = load_model('models/discriminator_model.h5')  # Change to your model path

def load_and_preprocess_image(img):
    # Load and preprocess the image
    img = img.resize((256, 256))  # Resize to match model input
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def predict_anomaly(img_array):
    score = model.predict(img_array)
    is_anomaly = score < 0.5  # Adjust threshold as needed
    return "Pneumonia" if is_anomaly else "Normal"

# Streamlit UI
st.title("Pneumonia Detection")
st.write("Upload an X-ray image to get a prediction.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the image
    img = image.load_img(uploaded_file)
    img_array = load_and_preprocess_image(img)

    # Make prediction
    result = predict_anomaly(img_array)

    # Display results
    st.image(img, caption='Uploaded Image', use_column_width=True)
    st.write(f"Prediction: {result}")
