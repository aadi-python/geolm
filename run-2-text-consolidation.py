import argparse
import os
import sys

# Ensure the package directory is in the Python path
# This assumes the script is run from the workspace root
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

try:
    # Use absolute import assuming the package is installed or in the path
    from hutton_lm.llm_interface import llm_consolidate_parsed_text
except ImportError:
    print("Error: Could not import hutton_lm package.")
    print(
        "Ensure the package is installed or the script is run from the correct directory."
    )
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Consolidate extracted geological text using an LLM."
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default="extracted-data/bingham-canyon/extracted_text.txt",
        help="Path to the input text file (output of parse-pdf).",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="extracted-data/bingham-canyon/consolidated-text-rev01.txt",
        help="Path to save the consolidated text output file.",
    )
    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file

    print(f"Reading extracted text from: {input_file}")
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            extracted_text = f.read()
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}")
        sys.exit(1)
    except IOError as e:
        print(f"Error reading file {input_file}: {e}")
        sys.exit(1)

    if not extracted_text.strip():
        print(f"Error: Input file {input_file} is empty.")
        sys.exit(1)

    print("Running LLM to consolidate text...")
    consolidated_text = llm_consolidate_parsed_text(extracted_text)

    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir:  # Check if output_dir is not empty (e.g., not just a filename)
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            print(f"Error creating output directory {output_dir}: {e}")
            sys.exit(1)

    print(f"Saving consolidated text to: {output_file}")
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(consolidated_text)
        print("Consolidation successful.")
    except IOError as e:
        print(f"Error writing to file {output_file}: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during saving: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
