# app.py

import os
import shutil
import uuid
import uvicorn
import asyncio
from typing import List

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

import cv2
import pytesseract
import pdf2image
import pdfplumber
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

# Set up paths for external tools if necessary
pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'  # Update as needed

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(title="Document Processing API")

# Allow CORS if needed (e.g., for web frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the pre-trained image captioning model globally to avoid reloading
model_name = "nlpconnect/vit-gpt2-image-captioning"
model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device)
feature_extractor = ViTImageProcessor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}


def preprocess_document(input_path, output_dir):
    """
    Convert PDF to images and preprocess them.
    """
    # Convert PDF to images
    images = pdf2image.convert_from_path(input_path)
    preprocessed_images = []
    for i, img in enumerate(images):
        # Convert to grayscale
        img = img.convert('L')
        # Convert to OpenCV format
        img_cv = np.array(img)
        # Save preprocessed image
        output_path = os.path.join(output_dir, f"page_{i+1}.png")
        cv2.imwrite(output_path, img_cv)
        preprocessed_images.append(output_path)
    return preprocessed_images


def advanced_ocr(image_paths):
    """
    Perform OCR on images to extract text.
    """
    extracted_text = ""
    custom_oem_psm_config = r'--oem 3 --psm 6'
    for image_path in image_paths:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img, config=custom_oem_psm_config)
        extracted_text += text + "\n"
    return extracted_text


def detect_tables(image_paths):
    """
    Detect and extract tables from images.
    """
    table_data = []
    for image_path in image_paths:
        page_tables = extract_tables_with_pdfplumber(image_path)
        table_data.extend(page_tables)
    return table_data


def extract_tables_with_pdfplumber(image_path):
    """
    Extract tables using pdfplumber.
    """
    tables = []
    # Convert image back to PDF temporarily
    temp_pdf = image_path.replace('.png', '.pdf')
    img = Image.open(image_path)
    img.convert('RGB').save(temp_pdf)
    with pdfplumber.open(temp_pdf) as pdf:
        for page in pdf.pages:
            page_tables = page.extract_tables()
            for table in page_tables:
                tables.append(table)
    os.remove(temp_pdf)
    return tables


def process_images(image_paths):
    """
    Generate descriptions for images.
    """
    captions = []
    for image_path in image_paths:
        img = Image.open(image_path).convert('RGB')
        pixel_values = feature_extractor(
            images=img, return_tensors="pt").pixel_values.to(device)
        output_ids = model.generate(pixel_values, **gen_kwargs)
        caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        captions.append((image_path, caption))
    return captions


def contextual_integration(text, tables, captions):
    """
    Integrate text, tables, and captions into Markdown content.
    """
    content = ""
    content += "# Extracted Text\n\n"
    content += text + "\n\n"
    if tables:
        content += "# Extracted Tables\n\n"
        for idx, table in enumerate(tables):
            content += f"## Table {idx + 1}\n\n"
            content += convert_table_to_markdown(table) + "\n\n"
    if captions:
        content += "# Image Descriptions\n\n"
        for image_path, caption in captions:
            image_name = os.path.basename(image_path)
            content += f"![Image]({image_name})\n\n"
            content += f"**Description**: {caption}\n\n"
    return content


def convert_table_to_markdown(table):
    """
    Convert table data to Markdown format.
    """
    md = ''
    headers = table[0]
    md += '| ' + ' | '.join(headers) + ' |\n'
    md += '| ' + ' | '.join(['---'] * len(headers)) + ' |\n'
    for row in table[1:]:
        md += '| ' + ' | '.join(row) + ' |\n'
    return md


def data_normalization(content, output_path):
    """
    Save content to a Markdown file.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)


@app.post("/process")
async def process_document(file: UploadFile = File(...)):
    """
    Endpoint to process an uploaded PDF document.
    """
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400, detail="Only PDF files are supported.")

    # Limit file size (e.g., 50 MB)
    file_size = await file.read()
    max_file_size = 50 * 1024 * 1024  # 50 MB
    if len(file_size) > max_file_size:
        raise HTTPException(
            status_code=400, detail="File size exceeds the 50 MB limit.")
    await file.seek(0)

    # Create a unique temporary directory for processing
    temp_dir = os.path.join("temp", str(uuid.uuid4()))
    os.makedirs(temp_dir, exist_ok=True)
    input_pdf_path = os.path.join(temp_dir, "input.pdf")

    # Save the uploaded file
    with open(input_pdf_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Step 1: Document Preprocessing
        preprocessed_images = await asyncio.to_thread(preprocess_document, input_pdf_path, temp_dir)

        # Step 2: Advanced OCR for Text Extraction
        extracted_text = await asyncio.to_thread(advanced_ocr, preprocessed_images)

        # Step 3: Table Detection and Extraction
        tables = await asyncio.to_thread(detect_tables, preprocessed_images)

        # Step 4: Image and Graph Processing
        captions = await asyncio.to_thread(process_images, preprocessed_images)

        # Step 5: Contextual Integration
        content = contextual_integration(extracted_text, tables, captions)

        # Step 6: Data Normalization and Formatting
        output_markdown_path = os.path.join(temp_dir, "output.md")
        data_normalization(content, output_markdown_path)

        # Prepare the response
        return FileResponse(
            path=output_markdown_path,
            filename="output.md",
            media_type='text/markdown'
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Clean up temporary files
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
