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

    args = parser.parse_args()

    print(f"Running Hutton LM Generator in '{args.input_mode}' mode.")

    geo_model = None
    project_name = "Hutton_LM_Model"
    structure_file_to_use = args.structural_defs_file  # Default unless LLM mode

    if args.input_mode == "default":
        print("Initializing model using default data...")
        geo_model = initialize_geomodel_with_tmp_files(project_name)
    elif args.input_mode == "file":
        print(f"Initializing model using files:")
        print(f"  Orientations: {args.orientations_file}")
        print(f"  Points: {args.points_file}")
        # File existence is checked implicitly by initialize_geomodel_from_files via read_file_content
        # or directly by gempy. We rely on that check.
        geo_model = initialize_geomodel_from_files(
            project_name, args.orientations_file, args.points_file
        )
    elif args.input_mode == "llm":
        print("Running LLM generation...")
        generated_files = run_llm_generation(
            args.prompt_type, args.temperature, args.llm_output_dir
        )
        if not generated_files:
            print("LLM generation failed. Exiting.")
            return
        gen_points_file, gen_orientations_file, gen_structure_file = generated_files

        print("Initializing model using LLM generated files...")
        geo_model = initialize_geomodel_from_files(
            project_name + "_LLM", gen_orientations_file, gen_points_file
        )
        structure_file_to_use = gen_structure_file  # Use the generated structure file
    else:
        # This case should be unreachable due to argparse choices
        print(f"Internal Error: Invalid input mode '{args.input_mode}'.")
        return

    if geo_model is None:
        print("Failed to initialize GeoModel. Exiting.")
        return

    # Define structural framework using the determined structure file
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

    # Compute and plot
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
