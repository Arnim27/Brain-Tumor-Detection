import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# Load models
model_paths = ["models/Alexenet.h5", "models/resnet50.h5", "models/unet.h5"]
models = [load_model(path) for path in model_paths]

# Class labels for brain tumor types
class_labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# Function to preprocess image
def preprocess_image(image):
    # Resize image to (128, 128) and convert to grayscale if needed
    image = image.resize((128, 128))
    image_array = np.array(image)

    if image_array.ndim == 2:  # Convert grayscale to RGB
        image_array = np.stack((image_array,)*3, axis=-1)

    image_array = image_array / 255.0  # Normalize to [0, 1]
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Function to analyze U-Net output and return a class prediction
def get_unet_class_prediction(mask):
    # Flatten the mask to get pixel values
    flat_mask = mask.flatten()

    # Example: Assuming segmentation classes are labeled as 0, 1, 2, 3
    glioma_count = np.sum(flat_mask == 0)
    meningioma_count = np.sum(flat_mask == 1)
    no_tumor_count = np.sum(flat_mask == 2)
    pituitary_count = np.sum(flat_mask == 3)

    # Get the predicted class based on maximum count
    counts = [glioma_count, meningioma_count, no_tumor_count, pituitary_count]
    predicted_class = int(np.argmax(counts))

    return predicted_class

# Function to perform majority voting
def predict_with_majority_voting(image_array):
    predictions = []

    for model in models:
        preds = model.predict(image_array)

        # Handle U-Net output separately
        if preds.shape == (1, 128, 128, 1):
            predicted_class = get_unet_class_prediction(preds[0])
        elif preds.ndim == 1:
            predicted_class = int(np.argmax(preds))
        elif preds.ndim == 2:
            predicted_class = int(np.argmax(preds[0]))
        else:
            raise ValueError(f"Unexpected prediction output shape: {preds.shape}")

        predictions.append(predicted_class)

    # Apply majority voting
    final_prediction = max(set(predictions), key=predictions.count)
    return class_labels[final_prediction]

# Streamlit UI
st.title("Brain Tumor Classification with Majority Voting")
st.write("Upload an MRI image to predict the type of brain tumor.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    image_array = preprocess_image(image)

    # Make predictions with majority voting
    with st.spinner("Classifying..."):
        result = predict_with_majority_voting(image_array)

    # Display the result
    st.success(f"Predicted Tumor Type: {result}")
