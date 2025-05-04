import argparse
import numpy as np
import traceback

# Use relative imports for package modules
from .data_loader import (
    DEFAULT_POINTS_FILE,
    DEFAULT_ORIENTATIONS_FILE,
    DEFAULT_STRUCTURE_FILE,
)
from .llm_interface import run_llm_generation
from .model_builder import (
    initialize_geomodel_with_tmp_files,
    initialize_geomodel_from_files,
    load_structural_definitions,
    define_structural_groups,
    compute_and_plot_model,
)


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

    args = parser.parse_args()

    print(f"Running Hutton LM Generator in '{args.input_mode}' mode.")

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
        # File existence is checked implicitly by initialize_geomodel_from_files via read_file_content
        # or directly by gempy. We rely on that check.
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
                    raise RuntimeError("LLM generation step failed to produce files.")
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
                    # Optional: time.sleep(1)
                # Loop continues to the next attempt

        # --- End Retry Loop --- #

        if not success:
            print(
                f"LLM generation and model building failed after {args.retry_attempts} attempts. Exiting."
            )
            return  # Exit main if all retries failed

        # If loop succeeded, structure_file_to_use is already set correctly
        # for the successful attempt, and geo_model is the successfully built model.
        # The main flow can continue, but the successful parts (define, compute, plot)
        # have already run inside the loop. We might want to prevent them running again.

        # To prevent re-running post-loop, we can structure it so the post-loop code
        # is only for non-LLM modes or only runs if the loop didn't succeed (which it shouldn't reach here).
        # Let's adjust: The successful LLM run completes everything including plotting.

    else:
        # This case should be unreachable due to argparse choices
        print(f"Internal Error: Invalid input mode '{args.input_mode}'.")
        return

    # If input mode was NOT 'llm', proceed with defining groups and plotting
    if args.input_mode != "llm":
        if geo_model is None:
            print("Failed to initialize GeoModel. Exiting.")
            return

        # Define structural framework using the specified or default structure file
        try:
            print(f"Loading structural definitions from: {structure_file_to_use}")
            structural_definitions = load_structural_definitions(structure_file_to_use)
            if structural_definitions is None:
                print("Failed to load structural definitions. Exiting.")
                return

            define_structural_groups(geo_model, structural_definitions)

        except KeyError as e:
            print(f"\nError defining structural groups: Key '{e}' not found.")
            print(
                "This often means a surface/series name in the structural definitions CSV"
            )
            print("does not match the names provided in the points/orientations data.")
            print(
                f"  Check names in: {structure_file_to_use} against points/orientations data."
            )
            if geo_model and hasattr(geo_model, "structural_frame"):
                print(
                    f"  Available elements parsed from input: {list(geo_model.structural_frame.structural_elements.keys())}"
                )
            return
        except ValueError as e:
            print(f"\nError during structural group definition: {e}")
            return
        except Exception as e:
            print(f"\nAn unexpected error occurred defining structural groups: {e}")
            traceback.print_exc()
            return

        # Compute and plot for non-LLM modes
        try:
            compute_and_plot_model(geo_model)
        except Exception as e:
            print(f"\nAn error occurred during model computation or plotting: {e}")
            traceback.print_exc()
            return

    print("Hutton LM script finished successfully.")


if __name__ == "__main__":
    # Set seed for reproducibility if the script is run directly
    np.random.seed(1515)
    main()
