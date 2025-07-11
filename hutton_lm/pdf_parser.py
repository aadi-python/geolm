import os
import tempfile
import PyPDF2
from tqdm import tqdm
from typing import Optional, BinaryIO


# https://github.com/meta-llama/llama-cookbook/blob/main/end-to-end-use-cases/NotebookLlama/Step-1%20PDF-Pre-Processing-Logic.ipynb
def validate_pdf(file_path: str) -> bool:
    if not os.path.exists(file_path):
        print(f"Error: File not found at path: {file_path}")
        return False
    if not file_path.lower().endswith(".pdf"):
        print("Error: File is not a PDF")
        return False
    return True


def extract_text_from_pdf(file_path: str, max_chars: int = -1) -> Optional[str]:
    # max_chars = -1 for no text length limit
    if not validate_pdf(file_path):
        return None

    try:
        with open(file_path, "rb") as file:
            # Create PDF reader object
            pdf_reader = PyPDF2.PdfReader(file)

            # Get total number of pages
            num_pages = len(pdf_reader.pages)
            print(f"Processing PDF with {num_pages} pages...")

            extracted_text = []
            total_chars = 0

            # Iterate through all pages
            for page_num in tqdm(range(num_pages)):
                # Extract text from page
                page = pdf_reader.pages[page_num]
                text = page.extract_text()

                # Check if adding this page's text would exceed the limit
                if max_chars != -1 and total_chars + len(text) > max_chars:
                    # Only add text up to the limit
                    remaining_chars = max_chars - total_chars
                    extracted_text.append(text[:remaining_chars])
                    print(f"Reached {max_chars} character limit at page {page_num + 1}")
                    break

                extracted_text.append(text)
                total_chars += len(text)
                # print(f"Processed page {page_num + 1}/{num_pages}")

            final_text = "\n".join(extracted_text)
            print(f"\nExtraction complete! Total characters: {len(final_text)}")
            return final_text

    except PyPDF2.errors.PdfReadError:  # More specific exception
        print("Error: Invalid or corrupted PDF file")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        return None


def extract_images_from_pdf(file_path: str, output_dir: str) -> None:
    """
    Placeholder function to extract images from a PDF.
    Currently does nothing but print a message.
    """
    if not validate_pdf(file_path):
        return None

    # print(f"TODO: Implement image extraction from {file_path} into {output_dir}")
    # Placeholder: In a real implementation, this would extract images
    # and save them to the output_dir, potentially returning a list of image paths.
    return None


def extract_text(uploaded_file: BinaryIO) -> Optional[str]:
    """Extracts text from an uploaded PDF file object."""
    if uploaded_file is None:
        return None

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        return extract_text_from_pdf(tmp_path)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
