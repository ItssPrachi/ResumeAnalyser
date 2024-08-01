import fitz  # PyMuPDF
import os


def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file.

    Args:
    pdf_path (str): Path to the PDF file.

    Returns:
    str: Extracted text from the PDF.
    """
    try:
        # Open the PDF file
        doc = fitz.open(pdf_path)
        text = ""

        # Iterate through each page
        for page in doc:
            # Extract text from the page
            text += page.get_text()

        return text.strip()
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {str(e)}")
        return ""


def process_all_pdfs(directory):
    """
    Process all PDFs in a directory.

    Args:
    directory (str): Path to the directory containing PDF files.

    Returns:
    list: List of tuples containing filename and extracted text.
    """
    extracted_data = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(directory, filename)
            text = extract_text_from_pdf(pdf_path)
            extracted_data.append((filename, text))
    return extracted_data


# Usage
raw_data_dir = "data/raw/"
extracted_data = process_all_pdfs(raw_data_dir)