import os
import pandas as pd
from paddleocr import PaddleOCR
import cv2
from tqdm import tqdm  
import pytesseract
import logging
def pytesseract_ocr(image_path, pytesseract):

    # Read the image using OpenCV
    image = cv2.imread(image_path)

    # Convert the image to RGB (required by pytesseract)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform OCR on the image using pytesseract
    config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'  # Only digits
    predicted_text = pytesseract.image_to_string(image_rgb, config=config)

    return predicted_text.strip()  # Remove any trailing spaces
def paddleocr_ocr(image_path, ocr):
    # Perform OCR on the image
    result = ocr.ocr(image_path)

    # Extract predicted text 
    predicted_text = ""
    if result and isinstance(result, list) and len(result) > 0:
        if result[0] and isinstance(result[0], list) and len(result[0]) > 0:
            for line in result[0]:
                if line and isinstance(line, list) and len(line) > 1:
                    text = str(line[1][0])  # Convert recognized text to string
                    predicted_text += text  # Concatenate all text found
    else:
        predicted_text = ""
    return predicted_text.strip()  # Remove any trailing spaces

if __name__ == "__main__":
    # Suppress PaddleOCR debug info
    logging.getLogger('ppocr').setLevel(logging.WARNING)
    # Initialize PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)
    # Set the path to the Tesseract executable
    pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

    # Path to the folder containing the images
    folder_path = "../../input/crop/ibc_number/"

    # List to store results
    results = []

    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Loop through all image files with a progress bar
    for filename in tqdm(image_files, desc="Processing Images"):
        file_path = os.path.join(folder_path, filename)
        
        # Check if the file is an image 
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg')): 
            pytesseract_pt = pytesseract_ocr(file_path, pytesseract=pytesseract)
            paddle_pt = paddleocr_ocr(file_path,ocr=ocr)

            # Add the filename and predicted text to the results list
            results.append({
                'filename': filename,
                'pytesseract_predicted_result': pytesseract_pt.strip(),  # Remove any trailing spaces
                'paddleocr_ocr_predicted_result': paddle_pt.strip()  # Remove any trailing spaces
            })

    # Convert results to a pandas DataFrame
    df = pd.DataFrame(results)

    # Display the DataFrame
    print(df)

    # Optionally, save the DataFrame to a CSV file
    df.to_csv('ocr_results_tesseract.csv', index=False)
    print("Results saved to 'ocr_results_tesseract.csv'")