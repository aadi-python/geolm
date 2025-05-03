"""
Refactored version of data-to-3d-model.py focusing on clarity and modularity.
"""

import gempy as gp
import gempy_viewer as gpv
import numpy as np
import os
import tempfile
import argparse
from datetime import datetime  # Import datetime
import re  # Import regex for parsing

# Attempt to import the Llama API client, handle import error gracefully
try:
    from llama_api_client import LlamaAPIClient, APIError  # Import APIError too
except ImportError:
    print("Warning: llama_api_client not installed. LLM mode will not be available.")
    LlamaAPIClient = None
    APIError = None


def input_points():
    return """
,X,Y,Z,X_r,Y_r,Z_r,surface,series,id,order_series
0,200.0,0.0,-200.0,0.3126,0.46885,0.7501,seafloor,seafloor_series,6,1
1,200.0,100.0,-200.0,0.3126,0.53135,0.7501,seafloor,seafloor_series,6,1
2,500.0,0.0,-200.0,0.5001,0.46885,0.7501,seafloor,seafloor_series,6,1
3,500.0,100.0,-200.0,0.5001,0.53135,0.7501,seafloor,seafloor_series,6,1
4,800.0,0.0,-200.0,0.6876,0.46885,0.7501,seafloor,seafloor_series,6,1
5,800.0,100.0,-200.0,0.6876,0.53135,0.7501,seafloor,seafloor_series,6,1
7,700.0,0.0,-450.0,0.6251,0.46885,0.59385,rock1,right_series,4,2
6,700.0,100.0,-450.0,0.6251,0.53135,0.59385,rock1,right_series,4,2
8,900.0,0.0,-450.0,0.7501,0.46885,0.59385,rock1,right_series,4,2
9,900.0,100.0,-450.0,0.7501,0.53135,0.59385,rock1,right_series,4,2
13,700.0,0.0,-700.0,0.6251,0.46885,0.4376,rock2,right_series,5,2
12,700.0,100.0,-700.0,0.6251,0.53135,0.4376,rock2,right_series,5,2
11,900.0,0.0,-700.0,0.7501,0.46885,0.4376,rock2,right_series,5,2
10,900.0,100.0,-700.0,0.7501,0.53135,0.4376,rock2,right_series,5,2
14,300.0,0.0,-200.0,0.3751,0.46885,0.7501,onlap_surface,onlap_series,3,3
15,300.0,100.0,-200.0,0.3751,0.53135,0.7501,onlap_surface,onlap_series,3,3
16,700.0,0.0,-1000.0,0.6251,0.46885,0.2501,onlap_surface,onlap_series,3,3
17,700.0,100.0,-1000.0,0.6251,0.53135,0.2501,onlap_surface,onlap_series,3,3
18,550.0,0.0,-500.0,0.53135,0.46885,0.5626,onlap_surface,onlap_series,3,3
19,550.0,100.0,-500.0,0.53135,0.53135,0.5626,onlap_surface,onlap_series,3,3
20,100.0,0.0,-400.0,0.2501,0.46885,0.6251,rock3,left_series,2,4
21,100.0,100.0,-400.0,0.2501,0.53135,0.6251,rock3,left_series,2,4
22,450.0,0.0,-1000.0,0.46885,0.46885,0.2501,rock3,left_series,2,4
23,450.0,100.0,-1000.0,0.46885,0.53135,0.2501,rock3,left_series,2,4
24,300.0,0.0,-500.0,0.3751,0.46885,0.5626,rock3,left_series,2,4
25,300.0,100.0,-500.0,0.3751,0.53135,0.5626,rock3,left_series,2,4
    """


def input_orientations():
    return """
,X,Y,Z,X_r,Y_r,Z_r,G_x,G_y,G_z,dip,azimuth,polarity,surface,series,id,order_series
0,200.0,50.0,-200.0,0.3126,0.5001,0.7501,1e-12,1e-12,1.000000000001,0.0,0.0,1.0,seafloor,seafloor_series,1,1
1,500.0,50.0,-200.0,0.5001,0.5001,0.7501,1e-12,1e-12,1.000000000001,0.0,0.0,1.0,seafloor,seafloor_series,1,1
2,800.0,50.0,-200.0,0.6876,0.5001,0.7501,1e-12,1e-12,1.000000000001,0.0,0.0,1.0,seafloor,seafloor_series,1,1
4,700.0,50.0,-450.0,0.6251,0.5001,0.59385,1e-12,1e-12,1.000000000001,0.0,0.0,1.0,rock1,right_series,2,2
3,900.0,50.0,-450.0,0.7501,0.5001,0.59385,1e-12,1e-12,1.000000000001,0.0,0.0,1.0,rock1,right_series,2,2
6,700.0,50.0,-700.0,0.6251,0.5001,0.4376,1e-12,1e-12,1.000000000001,0.0,0.0,1.0,rock2,right_series,3,2
5,900.0,50.0,-700.0,0.7501,0.5001,0.4376,1e-12,1e-12,1.000000000001,0.0,0.0,1.0,rock2,right_series,3,2
7,300.0,50.0,-200.0,0.3751,0.5001,0.7501,1e-12,1e-12,1.000000000001,0.0,0.0,1.0,onlap_surface,onlap_series,4,3
8,700.0,50.0,-1000.0,0.6251,0.5001,0.2501,1.000000000001,1.0000612323399569e-12,1.0000612323399569e-12,90.0,90.0,1.0,onlap_surface,onlap_series,4,3
9,550.0,50.0,-500.0,0.53135,0.5001,0.5626,0.766044443119978,1.0000469066937634e-12,0.6427876096875393,50.0,90.0,1.0,onlap_surface,onlap_series,4,3
10,100.0,50.0,-400.0,0.2501,0.5001,0.6251,1e-12,1e-12,1.000000000001,0.0,0.0,1.0,rock3,left_series,5,4
11,450.0,50.0,-1000.0,0.46885,0.5001,0.2501,1.000000000001,2.3054614646942163e-12,1.0000612323399569e-12,90.0,89.9999999999252,1.0,rock3,left_series,5,4
12,300.0,50.0,-500.0,0.3751,0.5001,0.5626,0.766044443119978,1.0000469066937634e-12,0.6427876096875393,49.99999999992522,89.9999999999252,1.0,rock3,left_series,5,4
    """


# --- LLM Helper Functions ---


def get_llm_prompt(prompt_type: str) -> str:
    """Builds and returns the appropriate LLM prompt string based on the type."""

    prompt_intro = "You are a helpful assistant for geological modeling. Please generate two completely new CSV datasets suitable for GemPy: surface points and orientations."
    prompt_format_suffix = """Ensure the output format is exactly CSV, with the correct columns as shown in the reference/examples.
Clearly separate the two datasets using '=== POINTS DATA ===' and '=== ORIENTATIONS DATA ===' markers.
Include the CSV data within markdown code blocks (```csv ... ```).

Now, generate the data, keeping the structure and headers consistent. Start with '=== POINTS DATA ==='."""

    prompt_task = ""
    prompt_examples = ""

    if prompt_type == "default":
        prompt_task = "Please generate two CSV datasets for GemPy: surface points and orientations.\nUse the following examples as a base, but introduce some small modifications like changing some dip/azimuth values, adding or removing a rock type/surface, or adjusting point coordinates slightly."
        prompt_examples = f"""Example Points Data:
```csv
{input_points().strip()}
```

Example Orientations Data:
```csv
{input_orientations().strip()}
```"""

    elif prompt_type == "random":
        prompt_task = "Please generate two completely new CSV datasets suitable for GemPy: surface points and orientations.\nInvent a plausible but random geological structure (e.g., folded layers, a simple fault, an intrusion). Do NOT use the example data provided below as a base, only use it for format reference.\nDefine at least 3-5 distinct surfaces/rock types.\nGenerate a reasonable number of points (15-30) and orientations (10-20) to define the structure."
        prompt_examples = f"""Points Data Format Reference (DO NOT COPY VALUES):
```csv
{input_points().strip()}
```

Orientations Data Format Reference (DO NOT COPY VALUES):
```csv
{input_orientations().strip()}
```"""

    else:
        print(
            f"Warning: Unknown prompt type '{prompt_type}'. Falling back to default prompt."
        )
        # Recursively call with 'default' to handle unknown types
        return get_llm_prompt("default")

    # Combine the sections
    full_prompt = f"""{prompt_intro}

{prompt_task}

{prompt_examples}

{prompt_format_suffix}"""

    return full_prompt


def initialize_llm():
    """Initialize the Llama API client."""
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
        client = LlamaAPIClient(
            api_key=api_key
        )  # Base URL might be optional depending on client
        return client
    except Exception as e:
        print(f"Error initializing Llama API client: {e}")
        return None


def generate_data_with_llm(client, prompt, temperature):
    """Calls the LLM API to generate data based on the prompt."""
    if not client or not APIError:  # Check if client or APIError is None
        print("Error: LLM Client or APIError not available.")
        return None
    try:
        print(f"Sending prompt to LLM (Temperature: {temperature})...")
        response = client.chat.completions.create(
            model="Llama-4-Maverick-17B-128E-Instruct-FP8",  # Or desired model
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,  # Use temperature argument
            # max_tokens=3000 # Consider re-adding if needed, maybe as arg
        )
        print("LLM response received.")
        print(response)  # Keep the print for the full object as per previous change
        return response
    except APIError as e:
        print(f"LLM API Error: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during LLM generation: {e}")
        return None


def parse_llm_response(llm_response_object):
    """Parses the LLM response object to extract points and orientations CSV data."""
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
            # Optionally log the object structure for debugging
            # print(f"LLM Response Object: {llm_response_object}")
            return None, None

    except AttributeError as e:
        print(f"Error accessing attributes in LLM response object: {e}")
        return None, None

    if not response_text:
        print("Error: No text content found in LLM response.")
        return None, None

    # Use regex to find the data blocks, allowing for potential markdown code fences
    # (?s) flag makes . match newlines
    points_match = re.search(
        r"(?s)=== POINTS DATA ===.*?```csv\n(.*?)\n```", response_text
    )
    if not points_match:  # Fallback without code fences
        points_match = re.search(
            r"(?s)=== POINTS DATA ===\n(.*?)(\n=== ORIENTATIONS DATA ===|\Z)",
            response_text,
        )

    orientations_match = re.search(
        r"(?s)=== ORIENTATIONS DATA ===.*?```csv\n(.*?)\n```", response_text
    )
    if not orientations_match:  # Fallback without code fences
        orientations_match = re.search(
            r"(?s)=== ORIENTATIONS DATA ===\n(.*?)\Z", response_text
        )

    points_csv = points_match.group(1).strip() if points_match else None
    orientations_csv = (
        orientations_match.group(1).strip() if orientations_match else None
    )

    if not points_csv or not orientations_csv:
        print(
            "Error: Could not parse both points and orientations data from LLM response text."
        )
        print(
            "Expected format markers: === POINTS DATA === and === ORIENTATIONS DATA ==="
        )
        # Optionally print the raw response text for debugging
        # print("\n--- LLM Raw Response Text ---\n")
        # print(response_text)
        # print("\n--- End Raw Response Text ---\n")
        return None, None

    return points_csv, orientations_csv


def save_generated_data(points_csv, orientations_csv, output_dir):
    """Saves the generated points and orientations data to timestamped files."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        points_filename = os.path.join(output_dir, f"points_{timestamp}.csv")
        orientations_filename = os.path.join(
            output_dir, f"orientations_{timestamp}.csv"
        )

        with open(points_filename, "w") as f:
            f.write(points_csv)
        print(f"Saved generated points data to: {points_filename}")

        with open(orientations_filename, "w") as f:
            f.write(orientations_csv)
        print(f"Saved generated orientations data to: {orientations_filename}")

        return points_filename, orientations_filename
    except OSError as e:
        print(f"Error creating directory or writing files: {e}")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred during saving: {e}")
        return None, None


# --- End LLM Helper Functions ---


# Functions to use Llama 4 API to generate input data - Now implemented indirectly


def generate_input_orientations_llm():
    """Generate input orientations using Llama 4 API. (Deprecated - use main LLM flow)"""
    pass


def generate_input_points_llm():
    """Generate input points using Llama 4 API. (Deprecated - use main LLM flow)"""
    pass


def initialize_geomodel_from_files(
    project_name: str, path_to_orientations: str, path_to_points: str
) -> gp.data.GeoModel:
    """Initializes the GemPy GeoModel with data and topography from files."""  # Added 'from files'
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


def initialize_geomodel_with_tmp_files(project_name: str) -> gp.data.GeoModel:
    """Initializes the GemPy GeoModel using hardcoded data written to temporary files."""  # Clarified docstring
    temp_file_path_orientations = None
    temp_file_path_points = None
    geo_model = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".csv"
        ) as tmp_file_orientations:
            tmp_file_orientations.write(input_orientations())
            temp_file_path_orientations = tmp_file_orientations.name

        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".csv"
        ) as tmp_file_points:
            tmp_file_points.write(input_points())
            temp_file_path_points = tmp_file_points.name

        # Pass the paths to the initialize function
        geo_model = initialize_geomodel_from_files(
            project_name, temp_file_path_orientations, temp_file_path_points
        )

    except Exception as e:
        print(f"Error during temporary file creation or model initialization: {e}")
        # Ensure cleanup happens even if initialization fails mid-way
    finally:
        # Clean up the temporary files
        if temp_file_path_points and os.path.exists(temp_file_path_points):
            os.remove(temp_file_path_points)
        if temp_file_path_orientations and os.path.exists(temp_file_path_orientations):
            os.remove(temp_file_path_orientations)

    # Return the model (or None if it failed)
    return geo_model


def define_structural_groups(geo_model: gp.data.GeoModel):
    """Defines the structural groups and relationships for the model."""
    gp.add_structural_group(
        model=geo_model,
        group_index=0,
        structural_group_name="seafloor_series",
        elements=[geo_model.structural_frame.get_element_by_name("seafloor")],
        structural_relation=gp.data.StackRelationType.ERODE,
    )

    gp.add_structural_group(
        model=geo_model,
        group_index=1,
        structural_group_name="right_series",
        elements=[
            geo_model.structural_frame.get_element_by_name("rock1"),
            geo_model.structural_frame.get_element_by_name("rock2"),
        ],
        structural_relation=gp.data.StackRelationType.ONLAP,
    )

    gp.add_structural_group(
        model=geo_model,
        group_index=2,
        structural_group_name="onlap_series",
        elements=[geo_model.structural_frame.get_element_by_name("onlap_surface")],
        structural_relation=gp.data.StackRelationType.ERODE,
    )

    gp.add_structural_group(
        model=geo_model,
        group_index=3,
        structural_group_name="left_series",
        elements=[geo_model.structural_frame.get_element_by_name("rock3")],
        structural_relation=gp.data.StackRelationType.BASEMENT,
    )

    gp.remove_structural_group_by_name(model=geo_model, group_name="default_formation")


def compute_and_plot_model(geo_model: gp.data.GeoModel):
    """Computes the model and generates the final 3D plot."""
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
    print("Script finished. Plot window should be open.")


# --- New LLM Orchestration Function ---


def run_llm_generation(prompt_type: str, temperature: float, output_dir: str):
    """Orchestrates the LLM data generation process."""
    client = initialize_llm()
    if not client:
        print("Exiting due to LLM initialization failure.")
        return None, None

    # Construct the prompt based on type
    prompt = get_llm_prompt(prompt_type)

    llm_response = generate_data_with_llm(client, prompt, temperature)
    if not llm_response:
        print("Failed to get response from LLM. Exiting.")
        return None, None

    points_csv, orientations_csv = parse_llm_response(llm_response)
    if not points_csv or not orientations_csv:
        print("Failed to parse LLM response. Exiting.")
        return None, None

    # --- Print generated data --- #
    print("\n--- Generated Points Data ---")
    print(points_csv)
    print("\n--- Generated Orientations Data ---")
    print(orientations_csv)
    print("\n-----------------------------\n")
    # ---------------------------- #

    generated_points_file, generated_orientations_file = save_generated_data(
        points_csv, orientations_csv, output_dir
    )

    if not generated_points_file or not generated_orientations_file:
        print("Failed to save generated data. Exiting.")
        return None, None

    return generated_points_file, generated_orientations_file


# --- End LLM Orchestration Function ---


def main():
    """Main execution function."""

    parser = argparse.ArgumentParser(
        description="Generate 3D geological model using GemPy."
    )
    parser.add_argument(
        "--input-mode",
        type=str,
        choices=["default", "file", "llm"],
        default="default",
        help="Input data source mode: 'default' (hardcoded), 'file' (CSV files), 'llm' (generate via LLM).",  # Updated help text
    )
    parser.add_argument(
        "--orientations-file",
        type=str,
        default="input-data/default/default_orientations.csv",
        help="Path to the orientations CSV file (used only if --input-mode=file).",
    )
    parser.add_argument(
        "--points-file",
        type=str,
        default="input-data/default/default_points.csv",
        help="Path to the surface points CSV file (used only if --input-mode=file).",
    )
    parser.add_argument(
        "--llm-output-dir",
        type=str,
        default="input-data/llm-generated",
        help="Directory to save LLM-generated input files (used only if --input-mode=llm).",
    )
    parser.add_argument(
        "--prompt-type",
        type=str,
        default="default",
        choices=["default", "random"],  # Add more choices as needed
        help="Type of prompt to use for LLM generation (used only if --input-mode=llm).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for LLM generation (used only if --input-mode=llm).",
    )

    args = parser.parse_args()

    print(f"Running in {args.input_mode} mode.")

    geo_model = None
    project_name = "Onlap_relations_CLI"

    if args.input_mode == "default":
        geo_model = initialize_geomodel_with_tmp_files(project_name)
    elif args.input_mode == "file":
        print(f"Using orientations file: {args.orientations_file}")
        print(f"Using points file: {args.points_file}")
        if not os.path.exists(args.orientations_file):
            print(f"Error: Orientations file not found: {args.orientations_file}")
            return
        if not os.path.exists(args.points_file):
            print(f"Error: Points file not found: {args.points_file}")
            return
        geo_model = initialize_geomodel_from_files(
            project_name, args.orientations_file, args.points_file
        )
    elif args.input_mode == "llm":
        generated_points_file, generated_orientations_file = run_llm_generation(
            args.prompt_type, args.temperature, args.llm_output_dir
        )

        if not generated_points_file or not generated_orientations_file:
            print("LLM generation failed. Exiting.")
            return

        # Use the newly generated files
        geo_model = initialize_geomodel_from_files(
            project_name + "_LLM", generated_orientations_file, generated_points_file
        )

    if geo_model is None:
        print("Failed to initialize GeoModel.")
        return

    # Define structural framework
    # Ensure the surfaces/series defined in the (potentially modified) data exist
    try:
        define_structural_groups(geo_model)
    except KeyError as e:
        print(f"\nError defining structural groups: {e}")
        print("This likely means the LLM generated data with surface/series names")
        print(
            "that don't match the expected names in the define_structural_groups function."
        )
        print(
            "Please check the generated files or adjust the define_structural_groups function."
        )
        return  # Stop execution if groups can't be defined
    except Exception as e:
        print(f"\nAn unexpected error occurred defining structural groups: {e}")
        return

    # Compute and plot
    compute_and_plot_model(geo_model)


if __name__ == "__main__":
    np.random.seed(1515)
    main()
