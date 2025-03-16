from ultralytics import YOLO
import cv2
import os
from tqdm import tqdm
from dotenv import dotenv_values

CONFIG = dotenv_values("../../.env")
MODEL_PATH = CONFIG["MODEL_PATH"]
MODEL_NAME = CONFIG["MODEL_NAME"]
MODEL = YOLO(f"{MODEL_PATH}/{MODEL_NAME}")

# Directory containing test images
test_images_dir = "../../input/images"
output_crops_dir = "../../input/crop"

# Create output directory if it doesn't exist
os.makedirs(output_crops_dir, exist_ok=True)
# List all image files in the test directory
image_files = os.listdir(test_images_dir)


# Use tqdm to display a progress bar
for image_file in tqdm(image_files, desc="Processing images", unit="image"):
    image_path = os.path.join(test_images_dir, image_file)
    
    # Read the image
    image = cv2.imread(image_path)
    
    # Perform prediction
    results = MODEL.predict(image_path, save=False, imgsz=640)
    
    # Process the results
    for result_idx, (box, cls) in enumerate(zip(results[0].boxes.xyxy, results[0].boxes.cls)):
        x1, y1, x2, y2 = map(int, box)  # Get bounding box coordinates
        class_index = int(cls)  # Convert class index to integer
        class_name = MODEL.names[class_index]  # Get the class name
        
        # Create a directory for the class if it doesn't exist
        class_dir = os.path.join(output_crops_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        # Crop the detected object
        cropped_image = image[y1:y2, x1:x2]
        
        # Save the cropped image in the class-specific directory
        cropped_image_path = os.path.join(
            class_dir, f"{os.path.splitext(image_file)[0]}_crop_{result_idx}.jpg"
        )
        cv2.imwrite(cropped_image_path, cropped_image)