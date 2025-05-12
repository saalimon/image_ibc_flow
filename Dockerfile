FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 libgl1 tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY requirements.txt requirements.lock ./

# Install uv and use it for package installation
RUN pip install uv
RUN uv pip install --system --no-cache -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["python", "app.py"]
