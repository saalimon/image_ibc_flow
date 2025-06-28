import gradio as gr
import subprocess
from model.ocr_model.predict_number import OCREngine
from model.ocr_model.crop_image import cropping
from PIL import Image
import os
import pandas as pd
import cv2
from utils.get_images import GetImages
import time
from datetime import datetime

latest_frame = None
running = True

TEST_URL = "rtsp://host.docker.internal:8554/local-loop"

ocr_engine = OCREngine()

# Initialize the GetImages class
def run_cropping():
    output = cropping()
    return output


def run_prediction_dir(
    folder_path="/app/input/crop/ibc_number/", progress=gr.Progress()
):
    if folder_path is not None:
        progress(0, desc="Processing Directory")
        OUTPUT_CSV_FILE = "ocr_results.csv"
        # Process the directory using the imported function
        results = ocr_engine.process_directory(folder_path, tqdm_obj=progress.tqdm)

        # Format the results
        output = "Prediction Output:\n"
        df = pd.DataFrame(results)
        df.to_csv(OUTPUT_CSV_FILE, index=False)

        print(f"Results saved to '{OUTPUT_CSV_FILE}'")
        for result in results:
            output += f"Path: {OUTPUT_CSV_FILE}\n"
            output += f"Length: {df.shape}\n"
        return output
    return "No directory uploaded."

def run_prediction(image):
    if image is not None:
        # Save the uploaded image
        image_path = "temp_image.png"
        image.save(image_path)

        # Process the image using the imported function
        results = ocr_engine.process_single_image(image_path)

        # Format the results
        output = "Prediction Output:\n"
        output += f"Pytesseract Result: {results['pytesseract_predicted_result']}\n"
        output += f"LevOCR Result: {results['levocr_predicted_result']}\n"
        output += f"TrOCR Result: {results['trocr_predicted_result']}"

        return output
    return "No image uploaded."


def train_model():
    # Run the training script
    result = subprocess.run(
        ["python", "model/ocr_model/train.py"], capture_output=True, text=True
    )
    return f"Training Output:\n{result.stdout}"


def get_rtsp_frame():
    # global latest_frame, running
    img_obj = GetImages("/app/.env")
    print(f"Trying to connect to: {img_obj.url}")
    cv2.setLogLevel(5)
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|timeout;5000000"
    cap = cv2.VideoCapture(img_obj.url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Set buffer size to 1 frame
    if not cap.isOpened():
        return None
    while running:
        ret, frame = cap.read()

        if not ret:
            return None
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        latest_frame = frame
        yield frame


# def capture_and_save():
#     frame = latest_frame
#     if frame is None:
#         return None, "No frame to capture."

#     # Save image to file
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     name = f"captured_frame_{timestamp}.jpg"
#     cv2.imwrite(f"/app/output/{name}", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
#     return frame, f"Captured and saved as {name}"


# Create the Gradio interface
with gr.Blocks(title="OCR Model Interface") as demo:
    gr.Markdown("# OCR Model Interface")

    with gr.Tab("Get Numbers"):
        get_numbers_button = gr.Button("Run Get Numbers")
        # capture_button = gr.Button("Capture Frame")
        output_video = gr.Image(label="RTSP Frame", type="numpy", live=True)
        get_numbers_button.click(fn=get_rtsp_frame, inputs=None, outputs=output_video)
        # capture_button.click(fn=capture_and_save, inputs=None, outputs=None)

    with gr.Tab("Run Croping"):
        crop_button = gr.Button("Start Croping")
        crop_output = gr.Textbox(label="Croping Output", lines=10)
        crop_button.click(fn=run_cropping, inputs=None, outputs=crop_output)

    with gr.Tab("Run Prediction"):
        with gr.Column():
            predict_button_dir = gr.Button("Run Directory Prediction")
            output_text_dir = gr.Textbox(label="Output", lines=10)
            predict_button_dir.click(
                fn=run_prediction_dir, inputs=None, outputs=output_text_dir
            )

            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(
                        type="pil", label="Upload an image for prediction"
                    )
                    predict_button = gr.Button("Run Prediction")
                with gr.Column():
                    output_text = gr.Textbox(label="Output", lines=10)
            predict_button.click(
                fn=run_prediction, inputs=[image_input], outputs=output_text
            )

    with gr.Tab("Train Model"):
        train_button = gr.Button("Start Training")
        train_output = gr.Textbox(label="Training Output", lines=10)
        train_button.click(fn=train_model, inputs=None, outputs=train_output)

# Launch the app
if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
