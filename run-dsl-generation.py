import argparse
import os
import sys

# Ensure the package directory is in the Python path
# This assumes the script is run from the workspace root
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

try:
    # Use absolute import assuming the package is installed or in the path
    from hutton_lm.llm_interface import llm_generate_dsl_summary
except ImportError:
    print("Error: Could not import hutton_lm package.")
    print(
        "Ensure the package is installed or the script is run from the correct directory."
    )
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Generate geological DSL from consolidated text using an LLM."
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default="extracted-data/bingham-canyon/consolidated-text-rev01.txt",
        help="Path to the input consolidated text file.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="extracted-data/bingham-canyon/geo-dsl-rev1.txt",
        help="Path to save the generated DSL output file.",
    )
    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file

    print(f"Reading consolidated text from: {input_file}")
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            consolidated_text = f.read()
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}")
        sys.exit(1)
    except IOError as e:
        print(f"Error reading file {input_file}: {e}")
        sys.exit(1)

    if not consolidated_text.strip():
        print(f"Error: Input file {input_file} is empty or contains only whitespace.")
        sys.exit(1)

    print("Running LLM to generate DSL...")
    dsl_output = llm_generate_dsl_summary(consolidated_text)

    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir:
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            print(f"Error creating output directory {output_dir}: {e}")
            sys.exit(1)

    print(f"Saving generated DSL to: {output_file}")
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(dsl_output)
        print("DSL generation successful.")
    except IOError as e:
        print(f"Error writing to file {output_file}: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during saving: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
