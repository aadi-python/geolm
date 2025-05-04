"""
Refactored version of data-to-3d-model.py focusing on clarity and modularity.
"""

import gempy as gp
import gempy_viewer as gpv
import numpy as np
import os
import tempfile
import argparse
from datetime import datetime
import re
import csv

# --- Global Constants for Default Data ---
DEFAULT_INPUT_DIR = "input-data/default"
DEFAULT_POINTS_FILE = os.path.join(DEFAULT_INPUT_DIR, "default_points.csv")
DEFAULT_ORIENTATIONS_FILE = os.path.join(DEFAULT_INPUT_DIR, "default_orientations.csv")
DEFAULT_STRUCTURE_FILE = os.path.join(DEFAULT_INPUT_DIR, "default_structure.csv")


# Helper function to read file content
def read_file_content(filepath: str) -> str | None:
    """Reads the entire content of a file into a string."""
    try:
        with open(filepath, "r") as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: Default data file not found: {filepath}")
        return None
    except Exception as e:
        print(f"Error reading default data file '{filepath}': {e}")
        return None


# Load default data into global constants
DEFAULT_POINTS_DATA = read_file_content(DEFAULT_POINTS_FILE)
DEFAULT_ORIENTATIONS_DATA = read_file_content(DEFAULT_ORIENTATIONS_FILE)
DEFAULT_STRUCTURE_DATA = read_file_content(DEFAULT_STRUCTURE_FILE)

# Exit if essential default data is missing
if (
    DEFAULT_POINTS_DATA is None
    or DEFAULT_ORIENTATIONS_DATA is None
    or DEFAULT_STRUCTURE_DATA is None
):
    print("Critical Error: Failed to load essential default data files. Exiting.")
    exit(1)
# -----------------------------------------

# Attempt to import the Llama API client, handle import error gracefully
try:
    from llama_api_client import LlamaAPIClient, APIError
except ImportError:
    print("Warning: llama_api_client not installed. LLM mode will not be available.")
    LlamaAPIClient = None
    APIError = None

# --- LLM Helper Functions ---


def get_llm_prompt(prompt_type: str) -> str:
    """Builds and returns the appropriate LLM prompt string based on the type."""

    prompt_intro = "You are a helpful assistant for geological modeling. Please generate three completely new CSV datasets suitable for GemPy: surface points, orientations, and structural group definitions."
    prompt_format_suffix = """Ensure the output format is exactly CSV, with the correct columns as shown in the reference/examples.
Clearly separate the three datasets using '=== POINTS DATA ===', '=== ORIENTATIONS DATA ===', and '=== STRUCTURE DATA ===' markers.
Include the CSV data within markdown code blocks (```csv ... ```).
For the structure data, use only the following relations: ERODE, ONLAP, BASEMENT.

Now, generate the data, keeping the structure and headers consistent. Start with '=== POINTS DATA ==='."""

    prompt_task = ""
    prompt_examples = ""

    if prompt_type == "default":
        prompt_task = "Please generate three CSV datasets for GemPy: surface points, orientations, and structural definitions.\nUse the following examples as a base, but introduce some small modifications like changing some dip/azimuth values, adding or removing a rock type/surface (and updating structure accordingly), or adjusting point coordinates slightly."
        prompt_examples = f"""Example Points Data:
```csv
{DEFAULT_POINTS_DATA.strip()}
```

Example Orientations Data:
```csv
{DEFAULT_ORIENTATIONS_DATA.strip()}
```

Example Structure Data:
```csv
{DEFAULT_STRUCTURE_DATA.strip()}
```"""

    elif prompt_type == "random":
        prompt_task = "Please generate three completely new CSV datasets suitable for GemPy: surface points, orientations, and structural group definitions.\nInvent a plausible but random geological structure (e.g., folded layers, a simple fault, an intrusion). Do NOT use the example data provided below as a base, only use it for format reference.\nDefine at least 3-5 distinct surfaces/rock types.\nGenerate a reasonable number of points (15-30) and orientations (10-20) to define the structure.\nEnsure the structural definitions reference only the surfaces/rock types defined in the points data and use only ERODE, ONLAP, or BASEMENT relations."
        prompt_examples = f"""Points Data Format Reference (DO NOT COPY VALUES):
```csv
{DEFAULT_POINTS_DATA.strip()}
```

Orientations Data Format Reference (DO NOT COPY VALUES):
```csv
{DEFAULT_ORIENTATIONS_DATA.strip()}
```

Structure Data Format Reference (DO NOT COPY VALUES):
```csv
{DEFAULT_STRUCTURE_DATA.strip()}
```"""

    else:
        print(
            f"Warning: Unknown prompt type '{prompt_type}'. Falling back to default prompt."
        )
        return get_llm_prompt("default")  # Recursive call with default

    # Combine the sections
    full_prompt = f"""{prompt_intro}

{prompt_task}

{prompt_examples}

{prompt_format_suffix}"""

    return full_prompt


def initialize_llm():
    """Initializes and returns the Llama API client."""
    if LlamaAPIClient is None:
        print(
            "Error: LlamaAPIClient is not available. Please install llama_api_client."
        )
        return None
    api_key = os.environ.get("LLAMA_API_KEY")
    if not api_key:
        print("Error: LLAMA_API_KEY environment variable not set.")
        return None
    try:
        client = LlamaAPIClient(api_key=api_key)
        return client
    except Exception as e:
        print(f"Error initializing Llama API client: {e}")
        return None


def generate_data_with_llm(client, prompt, temperature):
    """Calls the LLM API to generate data based on the prompt."""
    if not client or not APIError:
        print("Error: LLM Client or APIError not available.")
        return None
    try:
        print(f"Sending prompt to LLM (Temperature: {temperature})...")
        response = client.chat.completions.create(
            model="Llama-4-Maverick-17B-128E-Instruct-FP8",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        print("LLM response received.")
        print(response)  # Keep full response print for debugging
        return response
    except APIError as e:
        print(f"LLM API Error: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during LLM generation: {e}")
        return None


def parse_llm_response(llm_response_object):
    """Parses the LLM response object to extract points, orientations, and structure CSV data."""
    response_text = None
    try:
        # Extract the text content from the response object structure
        if (
            llm_response_object
            and hasattr(llm_response_object, "completion_message")
            and hasattr(llm_response_object.completion_message, "content")
            and hasattr(llm_response_object.completion_message.content, "text")
        ):
            response_text = llm_response_object.completion_message.content.text
        else:
            print("Error: Unexpected LLM response object structure.")
            return None, None, None

    except AttributeError as e:
        print(f"Error accessing attributes in LLM response object: {e}")
        return None, None, None

    if not response_text:
        print("Error: No text content found in LLM response.")
        return None, None, None

    # Use regex to find the data blocks, allowing for potential markdown code fences
    points_match = re.search(
        r"(?s)=== POINTS DATA ===.*?```csv\n(.*?)\n```", response_text
    )
    if not points_match:  # Fallback without code fences
        points_match = re.search(
            r"(?s)=== POINTS DATA ===\n(.*?)\n=== ORIENTATIONS DATA ===", response_text
        )

    orientations_match = re.search(
        r"(?s)=== ORIENTATIONS DATA ===.*?```csv\n(.*?)\n```", response_text
    )
    if not orientations_match:  # Fallback without code fences
        orientations_match = re.search(
            r"(?s)=== ORIENTATIONS DATA ===\n(.*?)\n=== STRUCTURE DATA ===",
            response_text,
        )

    structure_match = re.search(
        r"(?s)=== STRUCTURE DATA ===.*?```csv\n(.*?)\n```", response_text
    )
    if not structure_match:  # Fallback without code fences
        structure_match = re.search(
            r"(?s)=== STRUCTURE DATA ===\n(.*?)\Z", response_text
        )

    points_csv = points_match.group(1).strip() if points_match else None
    orientations_csv = (
        orientations_match.group(1).strip() if orientations_match else None
    )
    structure_csv = structure_match.group(1).strip() if structure_match else None

    if not points_csv or not orientations_csv or not structure_csv:
        print(
            "Error: Could not parse points, orientations, and/or structure data from LLM response text."
        )
        print(
            "Expected format markers: === POINTS DATA ===, === ORIENTATIONS DATA ===, === STRUCTURE DATA ==="
        )
        # print("\n--- LLM Raw Response Text ---\n") # Optional debug print
        # print(response_text)
        # print("\n--- End Raw Response Text ---\n")
        return None, None, None

    return points_csv, orientations_csv, structure_csv


def save_generated_data(points_csv, orientations_csv, structure_csv, output_dir):
    """Saves the generated points, orientations, and structure data to timestamped files."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        points_filename = os.path.join(output_dir, f"points_{timestamp}.csv")
        orientations_filename = os.path.join(
            output_dir, f"orientations_{timestamp}.csv"
        )
        structure_filename = os.path.join(output_dir, f"structure_{timestamp}.csv")

        with open(points_filename, "w") as f:
            f.write(points_csv)
        print(f"Saved generated points data to: {points_filename}")

        with open(orientations_filename, "w") as f:
            f.write(orientations_csv)
        print(f"Saved generated orientations data to: {orientations_filename}")

        with open(structure_filename, "w") as f:
            f.write(structure_csv)
        print(f"Saved generated structure data to: {structure_filename}")

        return points_filename, orientations_filename, structure_filename
    except OSError as e:
        print(f"Error creating directory or writing files: {e}")
        return None, None, None
    except Exception as e:
        print(f"An unexpected error occurred during saving: {e}")
        return None, None, None


# --- End LLM Helper Functions ---

# --- CSV Structural Definition Loading ---

# Mapping from CSV relation string to GemPy enum
RELATION_MAP = {
    "ERODE": gp.data.StackRelationType.ERODE,
    "ONLAP": gp.data.StackRelationType.ONLAP,
    "BASEMENT": gp.data.StackRelationType.BASEMENT,
}


def load_structural_definitions(filepath: str) -> list | None:
    """Loads structural group definitions from a CSV file."""
    definitions = []
    try:
        with open(filepath, "r", newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            required_columns = ["group_index", "group_name", "elements", "relation"]
            if not all(col in reader.fieldnames for col in required_columns):
                print(
                    f"Error: CSV file '{filepath}' is missing required columns: {required_columns}."
                )
                return None

            for row_num, row in enumerate(reader, start=2):  # start=2 for header row
                try:
                    index = int(row["group_index"].strip())
                    name = row["group_name"].strip()
                    elements_list = [
                        elem.strip()
                        for elem in row["elements"].split(",")
                        if elem.strip()
                    ]
                    relation_str = row["relation"].strip().upper()

                    if not name:
                        print(
                            f"Warning: Skipping row {row_num} due to empty group_name."
                        )
                        continue
                    if not elements_list:
                        print(
                            f"Warning: Skipping row {row_num} (group '{name}') due to empty elements list."
                        )
                        continue
                    if relation_str not in RELATION_MAP:
                        print(
                            f"Warning: Skipping row {row_num} (group '{name}') due to invalid relation '{row['relation']}'. Valid relations are: {list(RELATION_MAP.keys())}"
                        )
                        continue

                    relation_enum = RELATION_MAP[relation_str]

                    definitions.append(
                        {
                            "group_index": index,
                            "group_name": name,
                            "elements": elements_list,
                            "relation": relation_enum,
                        }
                    )
                except ValueError:
                    print(
                        f"Warning: Skipping row {row_num} due to invalid integer value for group_index ('{row['group_index']}')."
                    )
                    continue
                except KeyError as e:
                    print(f"Warning: Skipping row {row_num} due to missing column: {e}")
                    continue

    except FileNotFoundError:
        print(f"Error: Structural definitions file not found: {filepath}")
        return None
    except Exception as e:
        print(f"Error reading structural definitions file '{filepath}': {e}")
        return None

    if not definitions:
        print(f"Warning: No valid structural definitions loaded from '{filepath}'.")
        return None

    print(f"Loaded {len(definitions)} structural definitions from {filepath}.")
    return definitions


# --- End CSV Loading ---


# --- Deprecated LLM Functions --- #
def generate_input_orientations_llm():
    """Deprecated: Generate input orientations using Llama 4 API."""
    pass


def generate_input_points_llm():
    """Deprecated: Generate input points using Llama 4 API."""
    pass


# -------------------------------- #

# --- GemPy Model Initialization ---


def initialize_geomodel_from_files(
    project_name: str, path_to_orientations: str, path_to_points: str
) -> gp.data.GeoModel:
    """Initializes the GemPy GeoModel with data and topography from files."""
    geo_model: gp.data.GeoModel = gp.create_geomodel(
        project_name=project_name,
        extent=[-200, 1000, -500, 500, -1000, 0],
        resolution=[50, 50, 50],
        refinement=6,
        importer_helper=gp.data.ImporterHelper(
            path_to_orientations=path_to_orientations,
            path_to_surface_points=path_to_points,
        ),
    )
    gp.set_topography_from_random(grid=geo_model.grid, d_z=np.array([-600, -100]))
    geo_model.input_transform.apply_anisotropy(gp.data.GlobalAnisotropy.NONE)
    return geo_model


def initialize_geomodel_with_tmp_files(project_name: str) -> gp.data.GeoModel | None:
    """Initializes the GemPy GeoModel using default data written to temporary files."""
    temp_file_path_orientations = None
    temp_file_path_points = None
    geo_model = None
    try:
        # Create temporary files for default data
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".csv"
        ) as tmp_file_orientations:
            tmp_file_orientations.write(DEFAULT_ORIENTATIONS_DATA)
            temp_file_path_orientations = tmp_file_orientations.name

        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".csv"
        ) as tmp_file_points:
            tmp_file_points.write(DEFAULT_POINTS_DATA)
            temp_file_path_points = tmp_file_points.name

        # Initialize model using the temporary file paths
        geo_model = initialize_geomodel_from_files(
            project_name, temp_file_path_orientations, temp_file_path_points
        )

    except Exception as e:
        print(f"Error during temporary file creation or model initialization: {e}")
        # geo_model remains None
    finally:
        # Clean up the temporary files regardless of success/failure
        if temp_file_path_points and os.path.exists(temp_file_path_points):
            os.remove(temp_file_path_points)
        if temp_file_path_orientations and os.path.exists(temp_file_path_orientations):
            os.remove(temp_file_path_orientations)

    return geo_model  # Return the model (or None if initialization failed)


# ----------------------------------

# --- Model Definition and Computation ---


def define_structural_groups(geo_model: gp.data.GeoModel, structural_definitions: list):
    """Defines the structural groups and relationships based on loaded definitions."""
    if not structural_definitions:
        print("Error: No structural definitions provided. Cannot define groups.")
        raise ValueError("Structural definitions are required.")

    # # Clear any existing groups potentially created by the importer
    # existing_group_names = list(geo_model.structural_frame.structural_groups.keys())
    # if existing_group_names:
    #     print(f"Clearing existing/automatically created groups: {existing_group_names}")
    #     for group_name in existing_group_names:
    #         try:
    #             geo_model.structural_frame.delete_structural_group(group_name)
    #         except ValueError:
    #             print(f"Warning: Could not delete group '{group_name}'.")

    defined_groups = set()
    for definition in structural_definitions:
        group_index = definition["group_index"]
        group_name = definition["group_name"]
        element_names = definition["elements"]
        relation = definition["relation"]

        print(
            f"Defining group '{group_name}' (Index: {group_index}, Relation: {relation.name})..."
        )

        # Retrieve element objects from the model
        elements = []
        missing_elements = []
        for name in element_names:
            try:
                element = geo_model.structural_frame.get_element_by_name(name)
                elements.append(element)
            except ValueError:
                missing_elements.append(name)

        if missing_elements:
            print(
                f"  Error: Could not find elements for group '{group_name}': {missing_elements}"
            )
            print(
                f"  Available elements in model: {list(geo_model.structural_frame.structural_elements.keys())}"
            )
            raise ValueError(
                f"Missing elements for group '{group_name}'. Check input data and structural definition file."
            )

        if not elements:
            print(
                f"  Warning: No valid elements found for group '{group_name}'. Skipping."
            )
            continue

        try:
            gp.add_structural_group(
                model=geo_model,
                group_index=group_index,
                structural_group_name=group_name,
                elements=elements,
                structural_relation=relation,
            )
            defined_groups.add(group_name)
            print(
                f"  Successfully defined group '{group_name}' with elements: {[e.name for e in elements]}"
            )
        except Exception as e:
            print(f"  Error adding structural group '{group_name}' to GemPy model: {e}")
            raise e  # Re-raise to indicate failure

    gp.remove_structural_group_by_name(model=geo_model, group_name="default_formation")


def compute_and_plot_model(geo_model: gp.data.GeoModel):
    """Computes the GemPy model and generates the 3D plot."""
    print("Computing model...")
    gp.compute_model(gempy_model=geo_model)

    print("Generating 3D plot...")
    gpv.plot_3d(
        model=geo_model,
        show_surfaces=True,
        show_data=True,
        image=False,
        show_topography=True,
        kwargs_plot_structured_grid={"opacity": 0.2},
    )
    print("Plot window should be open.")


# ---------------------------------------

# --- LLM Orchestration Function ---


def run_llm_generation(prompt_type: str, temperature: float, output_dir: str):
    """Orchestrates LLM data generation (points, orientations, structure)."""
    client = initialize_llm()
    if not client:
        print("Exiting due to LLM initialization failure.")
        return None, None, None

    prompt = get_llm_prompt(prompt_type)

    llm_response = generate_data_with_llm(client, prompt, temperature)
    if not llm_response:
        print("Failed to get response from LLM. Exiting.")
        return None, None, None

    points_csv, orientations_csv, structure_csv = parse_llm_response(llm_response)
    if not points_csv or not orientations_csv or not structure_csv:
        print("Failed to parse LLM response. Exiting.")
        return None, None, None

    # --- Print generated data --- #
    print("\n--- Generated Points Data ---")
    print(points_csv)
    print("\n--- Generated Orientations Data ---")
    print(orientations_csv)
    print("\n--- Generated Structure Data ---")
    print(structure_csv)
    print("\n-----------------------------\n")
    # ---------------------------- #

    generated_files = save_generated_data(
        points_csv, orientations_csv, structure_csv, output_dir
    )

    if not all(generated_files):
        print("Failed to save one or more generated data files. Exiting.")
        return None, None, None

    return generated_files  # Returns (points_filename, orientations_filename, structure_filename)


# -----------------------------------


def main():
    """Main execution function: parses arguments, loads/generates data, builds model, plots."""

    parser = argparse.ArgumentParser(
        description="Generate 3D geological model using GemPy."
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
        default=DEFAULT_ORIENTATIONS_FILE,  # Use loaded default path
        help="Path to orientations CSV file (used if --input-mode=file).",
    )
    parser.add_argument(
        "--points-file",
        type=str,
        default=DEFAULT_POINTS_FILE,  # Use loaded default path
        help="Path to surface points CSV file (used if --input-mode=file).",
    )
    parser.add_argument(
        "--llm-output-dir",
        type=str,
        default="input-data/llm-generated",
        help="Directory to save LLM-generated input files (used if --input-mode=llm).",
    )
    parser.add_argument(
        "--structural-defs-file",
        type=str,
        default=DEFAULT_STRUCTURE_FILE,  # Use loaded default path
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

    print(f"Running in '{args.input_mode}' mode.")

    geo_model = None
    project_name = "Onlap_relations_CLI"
    structure_file_to_use = args.structural_defs_file  # Default unless LLM mode

    if args.input_mode == "default":
        geo_model = initialize_geomodel_with_tmp_files(project_name)
    elif args.input_mode == "file":
        print(f"Using orientations file: {args.orientations_file}")
        print(f"Using points file: {args.points_file}")
        if not os.path.exists(args.orientations_file) or not os.path.exists(
            args.points_file
        ):
            print("Error: One or both specified input files not found.")
            return
        geo_model = initialize_geomodel_from_files(
            project_name, args.orientations_file, args.points_file
        )
    elif args.input_mode == "llm":
        generated_files = run_llm_generation(
            args.prompt_type, args.temperature, args.llm_output_dir
        )
        if not generated_files:
            print("LLM generation failed. Exiting.")
            return
        gen_points_file, gen_orientations_file, gen_structure_file = generated_files

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
        print(
            f"  Available elements parsed from input: {list(geo_model.structural_frame.structural_elements.keys())}"
        )
        return
    except ValueError as e:
        print(f"\nError during structural group definition: {e}")
        return
    except Exception as e:
        print(f"\nAn unexpected error occurred defining structural groups: {e}")
        import traceback

        traceback.print_exc()
        return

    # Compute and plot
    compute_and_plot_model(geo_model)


if __name__ == "__main__":
    np.random.seed(1515)  # Set seed for reproducibility
    main()
