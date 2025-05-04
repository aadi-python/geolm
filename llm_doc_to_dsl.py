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


def get_llm_summary_dsl(consolidated_text: str) -> str:
    """
    Get the LLM response to the DSL prompt
    """

    # Convert consolidated text to prompt
    prompt_dsl = prompt_dsl_template.format(consolidated_text)

    client = OpenAI(
        api_key="LLM|1092127122939929|swnut7Dzo4N-CdXCmXFLKxWJC9s",
        base_url="https://api.llama.com/compat/v1/",
    )
    completion = client.chat.completions.create(
        model="Llama-4-Maverick-17B-128E-Instruct-FP8",
        messages=[{"role": "user", "content": prompt_dsl}],
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

# Prompt for the LLM to generate a DSL for the geological description
prompt_dsl_template = """
You are an expert geological interpreter and DSL compiler.  Your goal is to read a free-text description of an area's stratigraphy, structures, and events, then emit a concise, declarative DSL encoding of that description.

── INPUT ──

1) Geological description (condensed, one or two paragraphs):
{}

2) DSL specification and usage example:
```

# DSL Grammar

ROCK  <ID> [
name:    <string>;
type:    <sedimentary|volcanic|intrusive|metamorphic>;
age?:    <number><Ma|ka>|“?”;
]

DEPOSITION <ID> [
rock:  <ID>;
time?: <number><Ma|ka>|“?”;
after?: <ID>[, <ID>…];
]

EROSION <ID> [
time?: <number><Ma|ka>|“?”;
after?: <ID>[, <ID>…];
]

INTRUSION <ID> [
rock:    <ID>;
style?:  <dike|sill|stock|batholith>;
time?:   <number><Ma|ka>|“?”;
after?:  <ID>[, <ID>…];
]

# Example: Copper Porphyry System

ROCK   R1 [ name: Andesitic host;      type: volcanic;     age: 40Ma ]
ROCK   R2 [ name: Sedimentary cover;    type: sedimentary;  age: 38Ma ]
ROCK   R3 [ name: Quartz diorite porphyry; type: intrusive;  age: 35Ma ]
ROCK   R4 [ name: Copper-gold mineralization; type: intrusive;  age: 34Ma ]

DEPOSITION  D1 [ rock: R1;   time: 40Ma ]
DEPOSITION  D2 [ rock: R2;   time: 38Ma;   after: D1 ]

EROSION     E1 [ time: 37Ma; after: D2 ]

INTRUSION   I1 [ rock: R3;   style: stock;  time: 35Ma; after: D2, E1 ]
INTRUSION   I2 [ rock: R4;   style: dike;   time: 34Ma; after: I1 ]

```

── INSTRUCTIONS ──

• Parse the geological description.  
• Identify all distinct rock units, depositional events, erosional events, and intrusive events.  
• Assign a unique short ID to each (e.g. R1, D1, E1, I1…).  
• Fill in all known fields (`name`, `type`, `age`, `rock`, `time`, `style`, `after`).  
• Use absolute ages when given; otherwise use `after:` relationships to order events.  
• **Output ONLY** the DSL statements (one per line), in the order:  
  1. All `ROCK` definitions  
  2. All `DEPOSITION` statements  
  3. All `EROSION` statements  
  4. All `INTRUSION` statements  

Do **not** include any explanatory text—only the DSL code.
"""

# Insert the geological description into the prompt
prompt_dsl_filled = prompt_dsl_template.format(consolidated_text)

# Print the prompt
print(prompt_dsl_filled)

dsl_summary = get_llm_summary_dsl(consolidated_text)
print(dsl_summary)
