import argparse
import os
import sys

# Ensure the package directory is in the Python path
# This assumes the script is run from the workspace root
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

try:
    # Use absolute import assuming the package is installed or in the path
    from hutton_lm.pdf_parser import extract_text_from_pdf, extract_images_from_pdf
except ImportError:
    print("Error: Could not import hutton_lm package.")
    print(
        "Ensure the package is installed or the script is run from the correct directory."
    )
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Extract text and placeholder images from a PDF document."
    )
    parser.add_argument(
        "--input-pdf",
        type=str,
        default="assets/The_Bingham_Canyon_Porphyry_Cu_Mo_Au_Dep.pdf",
        help="Path to the input PDF file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="extracted-data/bingham-canyon",
        help="Directory to save the extracted text file and placeholder images.",
    )
    args = parser.parse_args()

    input_pdf = args.input_pdf
    output_dir = args.output_dir

    # Ensure output directory exists
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        print(f"Error creating output directory {output_dir}: {e}")
        sys.exit(1)

    # --- Text Extraction --- #
    print(f"Extracting text from: {input_pdf}")
    pdf_text = extract_text_from_pdf(input_pdf)

    if pdf_text:
        output_file_path = os.path.join(output_dir, "extracted_text-rev1.txt")
        print(f"Saving extracted text to: {output_file_path}")
        try:
            with open(output_file_path, "w", encoding="utf-8") as f:
                f.write(pdf_text)
            print("Text extraction successful.")
        except IOError as e:
            print(f"Error writing text file {output_file_path}: {e}")
            # Don't exit yet, still attempt image extraction
        except Exception as e:
            print(f"An unexpected error occurred during text saving: {e}")
    else:
        print(f"Failed to extract text from {input_pdf}. Skipping text saving.")
        # Continue to image extraction even if text fails

    # --- Placeholder Image Extraction --- #
    print("\nAttempting image extraction...")
    try:
        # Pass the validated output directory to the function
        extract_images_from_pdf(input_pdf, output_dir)
    except Exception as e:
        # Catch potential errors from image extraction if it were implemented
        print(f"An error occurred during placeholder image extraction: {e}")

    print("\nExtraction script finished.")


if __name__ == "__main__":
    main()
