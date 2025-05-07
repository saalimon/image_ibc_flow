import gradio as gr
import subprocess
import os
from model.ocr_model.predict_number import process_single_image

def run_prediction(image):
    if image is not None:
        # Save the uploaded image
        image_path = "temp_image.png"
        image.save(image_path)

        # Process the image using the imported function
        results = process_single_image(image_path)

        # Format the results
        output = "Prediction Output:\n"
        output += f"Pytesseract Result: {results['pytesseract_predicted_result']}\n"
        output += f"PaddleOCR Result: {results['paddleocr_ocr_predicted_result']}"

        return output
    return "No image uploaded."

def train_model():
    # Run the training script
    result = subprocess.run(["python", "model/ocr_model/train.py"], capture_output=True, text=True)
    return f"Training Output:\n{result.stdout}"

def get_numbers():
    # Run the get numbers script
    result = subprocess.run(["python", "utils/get_numbers.py"], capture_output=True, text=True)
    return f"Get Numbers Output:\n{result.stdout}"

# Create the Gradio interface
with gr.Blocks(title="OCR Model Interface") as demo:
    gr.Markdown("# OCR Model Interface")

    with gr.Tab("Run Prediction"):
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload an image for prediction")
                predict_button = gr.Button("Run Prediction")
            with gr.Column():
                output_text = gr.Textbox(label="Output", lines=10)
        predict_button.click(fn=run_prediction, inputs=image_input, outputs=output_text)

    with gr.Tab("Train Model"):
        train_button = gr.Button("Start Training")
        train_output = gr.Textbox(label="Training Output", lines=10)
        train_button.click(fn=train_model, inputs=None, outputs=train_output)

    with gr.Tab("Get Numbers"):
        get_numbers_button = gr.Button("Run Get Numbers")
        get_numbers_output = gr.Textbox(label="Get Numbers Output", lines=10)
        get_numbers_button.click(fn=get_numbers, inputs=None, outputs=get_numbers_output)

# Launch the app
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
