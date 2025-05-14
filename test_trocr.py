from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import requests
from PIL import Image
import time

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")

# load image from the IAM dataset
start_time = time.time()
url = "./temp_image.png"
image = Image.open(url).convert("RGB")

pixel_values = processor(image, return_tensors="pt").pixel_values
generated_ids = model.generate(pixel_values)

generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
elapsed_time = time.time() - start_time

print("Generated text:")
print(generated_text)
print(f"Time elapsed: {elapsed_time:.2f} seconds")
