import argparse
import os
import sys
import numpy as np
import traceback

# Use relative imports for package modules
from .data_loader import (
    DEFAULT_POINTS_FILE,
    DEFAULT_ORIENTATIONS_FILE,
    DEFAULT_STRUCTURE_FILE,
)
from .llm_interface import (
    run_llm_generation,
    llm_consolidate_parsed_text,
    llm_generate_dsl_summary,
)
from .model_builder import (
    initialize_geomodel_with_tmp_files,
    initialize_geomodel_from_files,
    load_structural_definitions,
    define_structural_groups,
    compute_and_plot_model,
)
from .pdf_parser import extract_text_from_pdf, extract_images_from_pdf

# Ensure the package directory is in the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def parse_pdf_command(args):
    """Handles the parse-pdf command."""
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    pdf_text = extract_text_from_pdf(args.input_pdf)
    if pdf_text:
        output_file_path = os.path.join(output_dir, "extracted_text.txt")
        try:
            with open(output_file_path, "w", encoding="utf-8") as f:
                f.write(pdf_text)
            print(f"Successfully extracted text to {output_file_path}")
        except IOError as e:
            print(f"Error writing to file {output_file_path}: {e}")

    else:
        print(f"Failed to extract text from {args.input_pdf}")
        sys.exit(1)  # Indicate failure

    # Call the placeholder image extraction function
    print("\nAttempting image extraction...")
    extract_images_from_pdf(args.input_pdf, output_dir)


def run_model_command(args):
    """Handles the run-model command."""
    print(f"Running model: {args.model_name}")
    print(f"Input data path: {args.input_data}")
    print(f"Output directory: {args.output_dir}")

    # Example usage (replace with actual logic)
    # loader = DataLoader(args.input_data)
    # data = loader.load()
    # builder = ModelBuilder(args.model_name)
    # model = builder.build()
    # results = model.run(data)
    # builder.save_output(results, args.output_dir)
    print("Model run simulation complete.")


def consolidate_text_command(args):
    """Handles the consolidate-text command."""
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

    print("Running LLM to consolidate text...")
    consolidated_text = llm_consolidate_parsed_text(extracted_text)

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"Saving consolidated text to: {output_file}")
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(consolidated_text)
        print("Consolidation successful.")
    except IOError as e:
        print(f"Error writing to file {output_file}: {e}")
        sys.exit(1)


def generate_dsl_command(args):
    """Handles the generate-dsl command."""
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

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"Saving generated DSL to: {output_file}")
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(dsl_output)
        print("DSL generation successful.")
    except IOError as e:
        print(f"Error writing to file {output_file}: {e}")
        sys.exit(1)


def main():
    """Main execution function: parses arguments, loads/generates data, builds model, plots."""

    parser = argparse.ArgumentParser(
        description="Generate 3D geological model using GemPy (Hutton LM Package)."
    )
    parser.add_argument(
        "--input-mode",
        type=str,
        choices=["default", "file", "llm"],
        default="default",
        help="Input data source: 'default', 'file' (requires --points-file, --orientations-file), or 'llm' (generates data).",
    )
    parser.add_argument(
        "--orientations-file",
        type=str,
        default=DEFAULT_ORIENTATIONS_FILE,
        help="Path to orientations CSV file (used if --input-mode=file).",
    )
    parser.add_argument(
        "--points-file",
        type=str,
        default=DEFAULT_POINTS_FILE,
        help="Path to surface points CSV file (used if --input-mode=file).",
    )
    parser.add_argument(
        "--llm-output-dir",
        type=str,
        default="input-data/llm-generated",  # Relative to workspace root by default
        help="Directory to save LLM-generated input files (used if --input-mode=llm).",
    )
    parser.add_argument(
        "--structural-defs-file",
        type=str,
        default=DEFAULT_STRUCTURE_FILE,
        help="Path to the structural definitions CSV file (used if --input-mode=file or default).",
    )
    parser.add_argument(
        "--prompt-type",
        type=str,
        default="default",
        choices=["default", "random"],
        help="Type of prompt for LLM generation (used if --input-mode=llm).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for LLM generation (used if --input-mode=llm).",
    )
    parser.add_argument(
        "--retry-attempts",
        type=int,
        default=5,
        help="Number of times to retry LLM generation and model building if an error occurs (used if --input-mode=llm).",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands (optional, runs GemPy generation if omitted)",
        required=False,
    )

    # --- Run Model Subcommand ---
    parser_run = subparsers.add_parser(
        "run-model", help="Run a specific geological model."
    )
    parser_run.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Name of the model to run (e.g., 'basin_model').",
    )
    parser_run.add_argument(
        "--input-data",
        type=str,
        required=True,
        help="Path to the input data file or directory.",
    )
    parser_run.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to save the model output.",
    )
    parser_run.set_defaults(func=run_model_command)

    # --- New PDF Parser Subcommand ---
    parser_parse_pdf = subparsers.add_parser(
        "parse-pdf", help="Extract text from a PDF document."
    )
    parser_parse_pdf.add_argument(
        "--input-pdf",
        type=str,
        default="assets/The_Bingham_Canyon_Porphyry_Cu_Mo_Au_Dep.pdf",
        help="Path to the input PDF file.",
    )
    parser_parse_pdf.add_argument(
        "--output-dir",
        type=str,
        default="extracted-data/bingham-canyon",
        help="Directory to save the extracted text file.",
    )
    parser_parse_pdf.set_defaults(func=parse_pdf_command)

    # --- New Text Consolidation Subcommand ---
    parser_consolidate = subparsers.add_parser(
        "consolidate-text", help="Consolidate extracted text using an LLM."
    )
    parser_consolidate.add_argument(
        "--input-file",
        type=str,
        default="extracted-data/bingham-canyon/extracted_text.txt",
        help="Path to the input text file (output of parse-pdf).",
    )
    parser_consolidate.add_argument(
        "--output-file",
        type=str,
        default="extracted-data/bingham-canyon/consolidated-text-rev01.txt",
        help="Path to save the consolidated text output file.",
    )
    parser_consolidate.set_defaults(func=consolidate_text_command)

    # --- New DSL Generation Subcommand ---
    parser_dsl = subparsers.add_parser(
        "generate-dsl",
        help="Generate geological DSL from consolidated text using an LLM.",
    )
    parser_dsl.add_argument(
        "--input-file",
        type=str,
        default="extracted-data/bingham-canyon/consolidated-text-rev01.txt",
        help="Path to the input consolidated text file.",
    )
    parser_dsl.add_argument(
        "--output-file",
        type=str,
        default="extracted-data/bingham-canyon/geo-dsl-rev1.txt",
        help="Path to save the generated DSL output file.",
    )
    parser_dsl.set_defaults(func=generate_dsl_command)

    args = parser.parse_args()

    # --- Command Execution --- #
    if args.command:
        # Execute the function associated with the chosen subcommand
        args.func(args)
        # Exit after running the specific command
        sys.exit(0)  # Use sys.exit(0) for clean exit
    else:
        # --- Original GemPy Generation Logic (if no subcommand is provided) --- #
        print(
            f"Running Hutton LM Generator in '{args.input_mode}' mode (no specific command given)."
        )

        geo_model = None
        project_name = "Hutton_LM_Model"
        structure_file_to_use = args.structural_defs_file  # Default unless LLM mode

        if args.input_mode == "default":
            print("Initializing model using default data...")
            geo_model = initialize_geomodel_with_tmp_files(project_name)
        elif args.input_mode == "file":
            print("Initializing model using files:")
            print(f"  Orientations: {args.orientations_file}")
            print(f"  Points: {args.points_file}")
            geo_model = initialize_geomodel_from_files(
                project_name, args.orientations_file, args.points_file
            )
        elif args.input_mode == "llm":
            success = False
            # --- Retry Loop --- #
            for attempt in range(args.retry_attempts):
                print(
                    f"\n--- LLM Generation Attempt {attempt + 1} of {args.retry_attempts} ---"
                )
                geo_model = None  # Reset geo_model for each attempt
                try:
                    print("Running LLM generation...")
                    generated_files = run_llm_generation(
                        args.prompt_type, args.temperature, args.llm_output_dir
                    )
                    if not generated_files:
                        raise RuntimeError(
                            "LLM generation step failed to produce files."
                        )
                    gen_points_file, gen_orientations_file, gen_structure_file = (
                        generated_files
                    )

                    print("Initializing model using LLM generated files...")
                    geo_model = initialize_geomodel_from_files(
                        project_name
                        + f"_LLM_Attempt_{attempt + 1}",  # Unique project name per attempt
                        gen_orientations_file,
                        gen_points_file,
                    )
                    if geo_model is None:
                        raise RuntimeError(
                            "Failed to initialize GeoModel from generated files."
                        )

                    # Use the generated structure file for this attempt
                    structure_file_to_use = gen_structure_file

                    print(
                        f"Loading structural definitions from generated file: {structure_file_to_use}"
                    )
                    structural_definitions = load_structural_definitions(
                        structure_file_to_use
                    )
                    if structural_definitions is None:
                        raise RuntimeError(
                            "Failed to load generated structural definitions."
                        )

                    print("Defining structural groups from generated definitions...")
                    define_structural_groups(geo_model, structural_definitions)

                    print("Computing and plotting model...")
                    compute_and_plot_model(geo_model)

                    # If all steps succeed:
                    print(f"--- Attempt {attempt + 1} successful! --- \n")
                    success = True
                    break  # Exit the retry loop on success

                except Exception as e:
                    print(f"--- Attempt {attempt + 1} failed: {e} --- \n")
                    # Optional: Log detailed traceback for debugging
                    # traceback.print_exc()

                    if attempt < args.retry_attempts - 1:
                        print("Retrying...")

            # --- End Retry Loop --- #

            if not success:
                print(
                    f"LLM generation and model building failed after {args.retry_attempts} attempts. Exiting."
                )
                sys.exit(1)  # Exit with error code

        else:
            # This case should be unreachable due to argparse choices
            print(f"Internal Error: Invalid input mode '{args.input_mode}'.")
            sys.exit(1)

        # If input mode was NOT 'llm' (and model exists), proceed with defining groups and plotting
        # For LLM mode, plotting already happened in the loop.
        if args.input_mode != "llm":
            if geo_model is None:
                print("Failed to initialize GeoModel. Exiting.")
                sys.exit(1)

            # Define structural framework using the specified or default structure file
            try:
                print(f"Loading structural definitions from: {structure_file_to_use}")
                structural_definitions = load_structural_definitions(
                    structure_file_to_use
                )
                if structural_definitions is None:
                    print("Failed to load structural definitions. Exiting.")
                    sys.exit(1)

                define_structural_groups(geo_model, structural_definitions)

            except KeyError as e:
                print(f"\nError defining structural groups: Key '{e}' not found.")
                print(
                    "This often means a surface/series name in the structural definitions CSV"
                )
                print(
                    "does not match the names provided in the points/orientations data."
                )
                print(
                    f"  Check names in: {structure_file_to_use} against points/orientations data."
                )
                if geo_model and hasattr(geo_model, "structural_frame"):
                    print(
                        f"  Available elements parsed from input: {list(geo_model.structural_frame.structural_elements.keys())}"
                    )
                sys.exit(1)
            except ValueError as e:
                print(f"\nError during structural group definition: {e}")
                sys.exit(1)
            except Exception as e:
                print(f"\nAn unexpected error occurred defining structural groups: {e}")
                traceback.print_exc()
                sys.exit(1)

            # Compute and plot for non-LLM modes
            try:
                compute_and_plot_model(geo_model)
            except Exception as e:
                print(f"\nAn error occurred during model computation or plotting: {e}")
                traceback.print_exc()
                sys.exit(1)

        print("Hutton LM script finished successfully.")


if __name__ == "__main__":
    # Set seed for reproducibility if the script is run directly
    np.random.seed(1515)
    main()
