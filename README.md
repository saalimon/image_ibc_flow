# image_ibc_flow
This application processes input images to extract and predict numbers using OCR (Optical Character Recognition).

## Command Line Usage
1. Add input images to the 'ibc_images' directory.
2. Run the 'crop_images.py' script located in the 'model/ocr_model' directory to crop the images.
3. Execute the 'predict_number.py' script to predict numbers from the cropped images.

Ensure that the input images are correctly placed in the 'ibc_images' directory before running the scripts.

## Web Interface
The application provides a web interface built with Gradio that allows you to:
- Upload images and run predictions
- Train the model
- Extract numbers from processed images

### Running the Web Interface
You can run the web interface using Docker:
```
docker-compose up gradio
```

Or directly with Python:
```
python app.py
```

The web interface will be available at http://localhost:7860
