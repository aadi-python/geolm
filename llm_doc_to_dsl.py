from openai import OpenAI
import PyPDF2
import os
import base64
from typing import Optional
import fitz
from tqdm import tqdm

# Functions


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

    except PyPDF2.PdfReadError:
        print("Error: Invalid or corrupted PDF file")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        return None


def llm_consolidate_parsed_text(pdf_text: str) -> str:
    """
    Consolidate the parsed text using the LLM
    """

    # Define the prompt text
    prompt = f"""
    You are a senior geologist with many years of experience.
    You are given a geological description of a region from a report.
    Your job is to consolidate the description into a single, coherent geological description.
    Focus on: rock types present, the orientation of large-scale structures, stratigraphic relationships, erosion, and igneous intrusions.
    Ignore: Chemical analysis, mineralogy, and other non-geological information.
    Condense the description into one or two paragraphs without line breaks, avoid using headings and bullet points.
    Here is the geological description:
    {pdf_text}
    """

    # Initialize the LLM client
    client = OpenAI(
        api_key="LLM|1092127122939929|swnut7Dzo4N-CdXCmXFLKxWJC9s",
        base_url="https://api.llama.com/compat/v1/",
    )
    completion = client.chat.completions.create(
        model="Llama-4-Maverick-17B-128E-Instruct-FP8",
        messages=[{"role": "user", "content": prompt}],
    )
    return completion.choices[0].message.content


# Step 1: Extract text from document

folder_path = "assets"
pdf_fp = os.path.join(folder_path, "The_Bingham_Canyon_Porphyry_Cu_Mo_Au_Dep.pdf")
pdf_text = extract_text_from_pdf(pdf_fp)
print(pdf_text)

# Step 1b: Extract images from document
# TODO

# Step 2: Text summary of extracted data

consolidated_text = llm_consolidate_parsed_text(pdf_text)
print(consolidated_text)

# Step 2b: Incorporate images into text summary

# Step 3
