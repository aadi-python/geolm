import gempy as gp
import gempy_viewer as gpv
import numpy as np
import os
import tempfile
import csv

# Use relative import for data constants within the package
from .data_loader import DEFAULT_POINTS_DATA, DEFAULT_ORIENTATIONS_DATA

# --- CSV Structural Definition Loading ---

# Mapping from CSV relation string to GemPy enum
RELATION_MAP = {
    "ERODE": gp.data.StackRelationType.ERODE,
    "ONLAP": gp.data.StackRelationType.ONLAP,
    "BASEMENT": gp.data.StackRelationType.BASEMENT,
}


def load_structural_definitions(filepath: str) -> list | None:
    """Loads structural group definitions from a CSV file."""
    # Use the file reader from data_loader to handle relative/absolute paths
    # Note: This reads the whole file, then parses. Could optimize if files are huge.
    # Alternatively, pass the absolute path logic from data_loader here.
    from .data_loader import read_file_content

    csv_content = read_file_content(filepath)
    if csv_content is None:
        return None  # Error already printed by read_file_content

    definitions = []
    try:
        # Use io.StringIO to treat the string content as a file for csv.DictReader
        import io

        csvfile = io.StringIO(csv_content)
        reader = csv.DictReader(csvfile)
        required_columns = ["group_index", "group_name", "elements", "relation"]

        # Check header
        if reader.fieldnames is None or not all(
            col in reader.fieldnames for col in required_columns
        ):
            print(
                f"Error: CSV file '{filepath}' is missing required columns: {required_columns}."
            )
            return None

        for row_num, row in enumerate(reader, start=2):  # start=2 for header row
            try:
                index = int(row["group_index"].strip())
                name = row["group_name"].strip()
                elements_list = [
                    elem.strip() for elem in row["elements"].split(",") if elem.strip()
                ]
                relation_str = row["relation"].strip().upper()

                if not name:
                    print(
                        f"Warning: Skipping row {row_num} in '{filepath}' due to empty group_name."
                    )
                    continue
                if not elements_list:
                    print(
                        f"Warning: Skipping row {row_num} in '{filepath}' (group '{name}') due to empty elements list."
                    )
                    continue
                if relation_str not in RELATION_MAP:
                    print(
                        f"Warning: Skipping row {row_num} in '{filepath}' (group '{name}') due to invalid relation '{row['relation']}'. Valid relations are: {list(RELATION_MAP.keys())}"
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
                    f"Warning: Skipping row {row_num} in '{filepath}' due to invalid integer value for group_index ('{row['group_index']}')."
                )
                continue
            except KeyError as e:
                print(
                    f"Warning: Skipping row {row_num} in '{filepath}' due to missing column: {e}"
                )
                continue

    except Exception as e:
        print(f"Error parsing structural definitions file '{filepath}': {e}")
        return None

    if not definitions:
        print(f"Warning: No valid structural definitions loaded from '{filepath}'.")
        return None

    print(f"Loaded {len(definitions)} structural definitions from {filepath}.")
    return definitions


# --- End CSV Loading ---

# --- GemPy Model Initialization ---


def initialize_geomodel_from_files(
    project_name: str, path_to_orientations: str, path_to_points: str
) -> gp.data.GeoModel:
    """Initializes the GemPy GeoModel with data and topography from files."""
    # Ensure paths are absolute or resolved relative to workspace root if needed
    # Assuming paths provided here are intended to be directly usable
    from .data_loader import _WORKSPACE_ROOT

    if not os.path.isabs(path_to_orientations):
        path_to_orientations = os.path.join(_WORKSPACE_ROOT, path_to_orientations)
    if not os.path.isabs(path_to_points):
        path_to_points = os.path.join(_WORKSPACE_ROOT, path_to_points)

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
        # These paths are absolute from NamedTemporaryFile
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

    # # Clear any existing groups potentially created by the importer (Commented out based on user edit)
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

    # Remove default formation (re-added based on user edit)
    try:
        gp.remove_structural_group_by_name(
            model=geo_model, group_name="default_formation"
        )
        print("Removed default_formation group if it existed.")
    except ValueError:
        pass  # Group didn't exist, that's fine


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


def compute_model_to_html(geo_model: gp.data.GeoModel, html_path: str) -> str:
    """Computes the model and exports an interactive HTML visualization."""
    import pyvista as pv
    import nest_asyncio

    nest_asyncio.apply()
    pv.start_xvfb()

    gp.compute_model(gempy_model=geo_model)
    vista = gpv.plot_3d(
        model=geo_model,
        show_surfaces=True,
        show_data=True,
        image=False,
        show_topography=True,
        kwargs_plot_structured_grid={"opacity": 0.2},
        show=False,
    )
    vista.p.export_html(html_path)
    return html_path


# ---------------------------------------
