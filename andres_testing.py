import os
import sys
import cv2
import pytesseract
import pdf2image
import pdfplumber
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer

# Set up paths for external tools if necessary
pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess_document(input_path, output_dir):
    """
    Step 1: Document Preprocessing
    Convert documents to a standard format (images) and enhance quality.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Convert PDF to images
    images = pdf2image.convert_from_path(input_path)
    preprocessed_images = []
    for i, img in enumerate(images):
        # Convert to grayscale
        img = img.convert('L')
        # Convert to OpenCV format
        img_cv = np.array(img)
        # Denoise and deskew (placeholder for actual implementation)
        # img_cv = cv2.fastNlMeansDenoising(img_cv, h=30)
        # Save preprocessed image
        output_path = os.path.join(output_dir, f"page_{i+1}.png")
        cv2.imwrite(output_path, img_cv)
        preprocessed_images.append(output_path)
    return preprocessed_images


def advanced_ocr(image_paths):
    """
    Step 2: Advanced OCR for Text Extraction
    Extract text from images with complex layouts.
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
    Step 3: Table Detection and Extraction
    Use deep learning models to detect and extract tables.
    """
    # Placeholder for table detection model
    # Let's assume we have a function `table_detector` that returns bounding boxes of tables
    table_data = []
    for image_path in image_paths:
        img = cv2.imread(image_path)
        # Placeholder for actual table detection
        # tables = table_detector(img)
        # For demonstration, we use pdfplumber as a fallback
        page_tables = extract_tables_with_pdfplumber(image_path)
        table_data.extend(page_tables)
    return table_data


def extract_tables_with_pdfplumber(image_path):
    """
    Fallback method to extract tables using pdfplumber.
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
    Step 4: Image and Graph Processing
    Classify and describe images using pre-trained models.
    """
    captions = []
    # Load pre-trained image captioning model
    model_name = "nlpconnect/vit-gpt2-image-captioning"
    model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device)
    feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    max_length = 16
    num_beams = 4
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

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
    Step 5: Contextual Integration
    Integrate extracted text, tables, and image captions.
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
            content += f"![Image]({image_path})\n\n"
            content += f"**Description**: {caption}\n\n"
    return content


def convert_table_to_markdown(table):
    """
    Convert a table (list of lists) to a Markdown table string.
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
    Step 6: Data Normalization and Formatting
    Save the content into a Markdown file suitable for LLMs.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)


def main():
    # Replace with your document path
    input_path = '/Users/matthewlew/Downloads/SampleContract-Shuttle.pdf'
    output_dir = '/Users/matthewlew/Downloads'
    output_markdown = '/Users/matthewlew/Downloads/output.md'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Step 1: Document Preprocessing
    print("Preprocessing document...")
    preprocessed_images = preprocess_document(input_path, output_dir)

    # Step 2: Advanced OCR for Text Extraction
    print("Performing OCR...")
    extracted_text = advanced_ocr(preprocessed_images)

    # Step 3: Table Detection and Extraction
    print("Detecting and extracting tables...")
    tables = detect_tables(preprocessed_images)

    # Step 4: Image and Graph Processing
    print("Processing images and generating descriptions...")
    captions = process_images(preprocessed_images)

    # Step 5: Contextual Integration
    print("Integrating content...")
    content = contextual_integration(extracted_text, tables, captions)

    # Step 6: Data Normalization and Formatting
    print("Saving content to Markdown...")
    data_normalization(content, output_markdown)

    print(f"Processing complete. Output saved to: {output_markdown}")


if __name__ == '__main__':
    main()
