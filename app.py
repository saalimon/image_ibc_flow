import streamlit as st
import subprocess

# Title
st.title("OCR Model Interface")

# Sidebar for navigation
st.sidebar.title("Navigation")
option = st.sidebar.selectbox("Choose an action", ["Run Prediction", "Train Model", "Get Numbers"])

# Run Prediction
if option == "Run Prediction":
    st.header("Run Prediction")
    input_image = st.file_uploader("Upload an image for prediction", type=["png", "jpg", "jpeg"])
    if input_image is not None:
        with open("temp_image.png", "wb") as f:
            f.write(input_image.getbuffer())
        st.write("Image uploaded successfully.")
        if st.button("Run Prediction"):
            result = subprocess.run(["python", "model/ocr_mode/predict_number.py"], capture_output=True, text=True)
            st.text("Prediction Output:")
            st.text(result.stdout)

# Train Model
elif option == "Train Model":
    st.header("Train Model")
    if st.button("Start Training"):
        with st.spinner("Training in progress..."):
            result = subprocess.run(["python", "model/ocr_mode/train.py"], capture_output=True, text=True)
            st.text("Training Output:")
            st.text(result.stdout)

# Get Numbers
elif option == "Get Numbers":
    st.header("Get Numbers")
    if st.button("Run Get Numbers"):
        result = subprocess.run(["python", "utils/get_numbers.py"], capture_output=True, text=True)
        st.text("Get Numbers Output:")
        st.text(result.stdout)