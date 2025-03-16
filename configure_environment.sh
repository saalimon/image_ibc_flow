#!/bin/bash
set -e
# Install OpenGL library
sudo apt-get update && sudo apt-get install -y libgl1-mesa-glx && apt-get -y install tesseract-ocr

pip install --upgrade pip && pip install -r ~/work/requirements.txt