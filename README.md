pip install these dependencies: 

pip install fastapi uvicorn[standard] pytesseract pdf2image pdfplumber opencv-python pillow torch torchvision transformers


Install tesseract OCR using:

brew install tesseract

Same with poppler: 

brew install poppler


Use api/interact with it using a curl command:

curl -X POST "http://localhost:8000/process" -F "file=@/Users/matthewlew/Downloads/SampleContract-Shuttle (1).pdf" --output ouput1.md


where: /Users/matthewlew/Downloads/SampleContract-Shuttle (1).pdf is just a file name



