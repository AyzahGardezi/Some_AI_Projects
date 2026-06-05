import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,    # pretrained model
    preprocess_input,
    decode_predictions
) 

from PIL import Image

def load_model():
    model = MobileNetV2(weights="imagenet")
    return model

def preprocess_image(image):
    # convert image to correct format
    img = np.array(image)
    img = cv2.resize(img, (224, 224))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)   # add another dimension to the image to create multiple images (model's requirement for multiple images)
    return img

def classify_image(model, image):
    try:
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        decoded_predictions = decode_predictions(predictions, top=3)[0]  # multiple predictions so we take the top 3, 0 index for the first image we sent, we receive a string
        return decoded_predictions
    except Exception as e:
        st.error(f"Error classifying image: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="AI Image Classifier", page_icon="🦦", layout="centered")
    st.title("AI Image Classifier")
    st.write("Upload an image so AI can guess what it is!")

    # only load the model the first time, then cache it (needed as streamlit reruns the script)
    @st.cache_resource
    def load_cached_model():
        return load_model()

    model = load_cached_model()

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png"])

    # display uploaded image
    if uploaded_file is not None:
        image = st.image(
            uploaded_file, caption="Uploaded Image", use_container_width=True
        )
        btn = st.button("Classify Image")

        if btn:
            with st.spinner("Analyzing Image"):
                image = Image.open(uploaded_file)
                predictions = classify_image(model, image)

                if predictions:
                    st.subheader("Predictions")
                    for _, label, score in predictions:
                        st.write(f"**{label}**: {score:.2%}")

if __name__ == "__main__":
    main()
