
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image
import time

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Start timing - loading image
start_time = time.time()
image = load_image("https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg")
image_load_time = time.time() - start_time
print(f"Image loading time: {image_load_time:.2f} seconds")

# Start timing - model loading
start_time = time.time()
processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
model = AutoModelForVision2Seq.from_pretrained(
    "HuggingFaceTB/SmolVLM-256M-Instruct",
    torch_dtype=torch.bfloat16,
    _attn_implementation="flash_attention_2" if DEVICE == "cuda" else "eager",
).to(DEVICE)
model_load_time = time.time() - start_time
print(f"Model loading time: {model_load_time:.2f} seconds")

# Create input messages
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Can you describe this image?"}
        ]
    },
]

# Start timing - input processing
start_time = time.time()
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=[image], return_tensors="pt")
inputs = inputs.to(DEVICE)
processing_time = time.time() - start_time
print(f"Input processing time: {processing_time:.2f} seconds")

# Start timing - generation
start_time = time.time()
print("Starting generation...")
generated_ids = model.generate(
    **inputs,
    max_length=256,  # Limit output length
    max_new_tokens=256,  # This controls only the new tokens generated
    num_beams=2,     # Reduce beam search complexity
    temperature=0.0,  # Control randomness
)

generation_time = time.time() - start_time
print(f"Generation time: {generation_time:.2f} seconds")

# Start timing - decoding
start_time = time.time()
generated_texts = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
)
decoding_time = time.time() - start_time
print(f"Decoding time: {decoding_time:.2f} seconds")

# Print total time
total_time = image_load_time + model_load_time + processing_time + generation_time + decoding_time
print(f"\nTotal execution time: {total_time:.2f} seconds")

print("\nGenerated text:")
print(generated_texts[0])

# Start timing - decoding
start_time = time.time()
print("Starting generation...")
generated_ids = model.generate(**inputs)
generation_time = time.time() - start_time
print(f"Generation time: {generation_time:.2f} seconds")

generated_texts = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
)
decoding_time = time.time() - start_time
print(f"[2] Decoding time: {decoding_time:.2f} seconds")

# Print total time
total_time = image_load_time + model_load_time + processing_time + generation_time + decoding_time
print(f"\n[2] Total execution time: {total_time:.2f} seconds")

print("\n[2] Generated text:")
print(generated_texts[0])