FROM python:3.8-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgomp1 libgl1 tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["python", "app.py"]
