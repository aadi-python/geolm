import argparse
import os
import sys
import numpy as np
import traceback

# Ensure the package directory is in the Python path
# This assumes the script is run from the workspace root
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

try:
    # Import necessary functions from the hutton_lm package
    from hutton_lm.pdf_parser import extract_text_from_pdf, extract_images_from_pdf
    from hutton_lm.llm_interface import (
        llm_consolidate_parsed_text,
        llm_generate_dsl_summary,
    )
    from hutton_lm.cli import run_core_gempy_workflow  # Import the refactored function

    # Also import defaults needed for argparse
    from hutton_lm.data_loader import (
        DEFAULT_POINTS_FILE,
        DEFAULT_ORIENTATIONS_FILE,
        DEFAULT_STRUCTURE_FILE,
    )
except ImportError as e:
    print(f"Error: Could not import hutton_lm package components: {e}")
    print(
        "Ensure the package is installed or the script is run from the correct directory."
    )
    sys.exit(1)


def main():
    # --- Argument Parsing ---
    # Combine arguments from all steps, providing reasonable defaults
    parser = argparse.ArgumentParser(
        description="Run the complete Hutton LM workflow: PDF -> Text -> Consolidation -> DSL -> GemPy Model."
    )

    # Step 1 Args
    parser.add_argument(
        "--input-pdf",
        type=str,
        default="assets/The_Bingham_Canyon_Porphyry_Cu_Mo_Au_Dep.pdf",
        help="Path to the input PDF file for the workflow.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="workflow-output/bingham-canyon",  # Default output dir for the whole workflow
        help="Base directory to save all intermediate and final workflow outputs.",
    )

    # Step 4 (GemPy) Args - Mirroring hutton_lm.cli main args
    gempy_group = parser.add_argument_group("GemPy Modeling Options (Step 4)")
    gempy_group.add_argument(
        "--input-mode",
        type=str,
        choices=["default", "file", "llm"],
        default="default",
        help="Input data source for GemPy modeling.",
    )
    gempy_group.add_argument(
        "--orientations-file",
        type=str,
        default=DEFAULT_ORIENTATIONS_FILE,
        help="Path to orientations CSV file (used if --input-mode=file).",
    )
    gempy_group.add_argument(
        "--points-file",
        type=str,
        default=DEFAULT_POINTS_FILE,
        help="Path to surface points CSV file (used if --input-mode=file).",
    )
    gempy_group.add_argument(
        "--llm-output-dir",
        type=str,
        default="input-data/llm-generated",  # Note: This is relative to workspace root in cli.py
        help="Directory for LLM-generated GemPy inputs (used if --input-mode=llm).",
    )
    gempy_group.add_argument(
        "--structural-defs-file",
        type=str,
        default=DEFAULT_STRUCTURE_FILE,
        help="Path to structural definitions CSV (used if --input-mode=file/default).",
    )
    gempy_group.add_argument(
        "--prompt-type",
        type=str,
        default="default",
        choices=["default", "random", "DSL"],
        help="Prompt type for LLM-based GemPy input generation.",
    )
    gempy_group.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="LLM temperature for GemPy input generation.",
    )
    gempy_group.add_argument(
        "--retry-attempts",
        type=int,
        default=15,
        help="Retry attempts for LLM-based GemPy input generation.",
    )

    args = parser.parse_args()

    # --- Setup ---
    base_output_dir = args.output_dir
    try:
        os.makedirs(base_output_dir, exist_ok=True)
        print(f"Workflow output directory: {base_output_dir}")
    except OSError as e:
        print(f"Error creating base output directory {base_output_dir}: {e}")
        sys.exit(1)

    # Define intermediate file paths
    extracted_text_file = os.path.join(base_output_dir, "extracted_text.txt")
    consolidated_text_file = os.path.join(base_output_dir, "consolidated_text.txt")
    dsl_file = os.path.join(base_output_dir, "geo_dsl.txt")

    pdf_text = None  # Initialize variable

    # Set numpy seed for reproducibility in GemPy step
    np.random.seed(1515)

    # --- Workflow Execution ---
    try:
        # --- Step 1: Text and Image Extraction ---
        print("\n--- Step 1: Extracting Text and Images ---")
        print(f"Input PDF: {args.input_pdf}")
        pdf_text = extract_text_from_pdf(args.input_pdf)  # Store extracted text

        if pdf_text:
            print(f"Saving extracted text to: {extracted_text_file}")
            with open(extracted_text_file, "w", encoding="utf-8") as f:
                f.write(pdf_text)
            print("Text extraction successful.")
        else:
            print(
                f"Warning: Failed to extract text from {args.input_pdf}. Attempting to continue..."
            )
            # Allow continuation, subsequent steps might fail gracefully if they need text

        print("Running image extraction...")
        extract_images_from_pdf(
            args.input_pdf, base_output_dir
        )  # Images go to base dir
        print("Step 1 finished.")

        # --- Step 2: Text Consolidation ---
        print("\n--- Step 2: Consolidating Text and Images ---")
        if not pdf_text or not pdf_text.strip():
            print(
                "Skipping text consolidation because no text was extracted or text is empty."
            )
            consolidated_text = None  # Ensure it's None if skipped
        else:
            print(f"Input text file (internal): {extracted_text_file}")
            print("Running LLM to consolidate text...")
            consolidated_text = llm_consolidate_parsed_text(pdf_text)
            if "Error:" in consolidated_text:
                print(f"LLM Consolidation Error: {consolidated_text}")
                # Decide if workflow should stop. Let's allow it to continue for now.
            else:
                print(f"Saving consolidated text to: {consolidated_text_file}")
                with open(consolidated_text_file, "w", encoding="utf-8") as f:
                    f.write(consolidated_text)
                print("Text consolidation successful.")
        print("Step 2 finished.")

        # --- Step 3: DSL Generation ---
        print("\n--- Step 3: Generating Geology DSL ---")
        if not consolidated_text or not consolidated_text.strip():
            print(
                "Skipping DSL generation because consolidated text is missing or empty."
            )
        else:
            print(f"Input consolidated text file (internal): {consolidated_text_file}")
            print("Running LLM to generate DSL...")
            dsl_output = llm_generate_dsl_summary(consolidated_text)
            if "Error:" in dsl_output:
                print(f"LLM DSL Generation Error: {dsl_output}")
                # Decide if workflow should stop
            else:
                print(f"Saving generated DSL to: {dsl_file}")
                with open(dsl_file, "w", encoding="utf-8") as f:
                    f.write(dsl_output)
                print("DSL generation successful.")
        print("Step 3 finished.")

        # --- Step 4: GemPy Modeling ---
        print("\n--- Step 4: Running Geology DSL to GemPy Modeling Workflow ---")
        # Pass the relevant arguments (parsed earlier) to the core workflow function
        gempy_success = run_core_gempy_workflow(args)
        if gempy_success:
            print("Step 4 (GemPy Modeling) finished successfully.")
        else:
            print("Step 4 (GemPy Modeling) failed.")
            # Optionally exit with error
            # sys.exit(1)

    except Exception as e:
        print(f"\n--- Workflow Interrupted by Error ---")
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()
        sys.exit(1)

    print("\n--- Full Workflow Finished ---")


if __name__ == "__main__":
    main()
