import os
import logging
from typing import Dict, List
import easyocr
import pandas as pd
from paddleocr import PaddleOCR
import cv2
from tqdm import tqdm
import pytesseract

# Configuration constants
TESSERACT_PATH = '/usr/bin/tesseract'
SUPPORTED_IMAGE_FORMATS = ('.png', '.jpg', '.jpeg')
TESSERACT_CONFIG = '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'
OUTPUT_CSV_FILE = 'ocr_results_tesseract.csv'
ALLOWED_DIGITS = '0123456789'


class OCREngine:
    def __init__(self):
        self._setup_logging()
        self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)
        self.easy_reader = easyocr.Reader(['en'])
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

    def _setup_logging(self) -> None:
        logging.getLogger('ppocr').setLevel(logging.WARNING)

    def _preprocess_image(self, image_path: str):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def process_with_tesseract(self, image_path: str) -> str:
        image_rgb = self._preprocess_image(image_path)
        return pytesseract.image_to_string(
            image_rgb,
            config=TESSERACT_CONFIG
        ).strip()

    def process_with_paddle(self, image_path: str) -> str:
        result = self.paddle_ocr.ocr(image_path, det=True, rec=True, cls=True)
        if not result or not isinstance(result, list) or not result[0]:
            return ""

        predicted_text = ""
        for line in result[0]:
            if line and isinstance(line, list) and len(line) > 1:
                text = str(line[1][0])
                predicted_text += ''.join(char for char in text if char.isdigit())
        return predicted_text.strip()

    def process_with_easyocr(self, image_path: str) -> str:
        text_results = self.easy_reader.readtext(image_path, allowlist=ALLOWED_DIGITS)
        return ''.join(text[1] for text in text_results).strip()

    def process_single_image(self, image_path: str) -> Dict[str, str]:
        return {
            'pytesseract_predicted_result': self.process_with_tesseract(image_path),
            'paddleocr_ocr_predicted_result': self.process_with_paddle(image_path),
            'easyocr_predicted_result': self.process_with_easyocr(image_path)
        }

    def process_directory(self, folder_path: str) -> List[Dict[str, str]]:
        results = []
        image_files = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith(SUPPORTED_IMAGE_FORMATS)
        ]

        for filename in tqdm(image_files, desc="Processing Images"):
            file_path = os.path.join(folder_path, filename)
            try:
                ocr_results = self.process_single_image(file_path)
                ocr_results['filename'] = filename
                results.append(ocr_results)
            except Exception as e:
                logging.error(f"Error processing {filename}: {str(e)}")

        return results


def main():
    ocr_engine = OCREngine()
    try:
        results = ocr_engine.process_directory("./")
        df = pd.DataFrame(results)
        df.to_csv(OUTPUT_CSV_FILE, index=False)
        print(f"Results saved to '{OUTPUT_CSV_FILE}'")
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")


if __name__ == "__main__":
    main()