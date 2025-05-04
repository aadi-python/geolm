import os
import sys

# --- Global Constants for Default Data ---
# Determine the absolute path to the package directory to reliably find 'input-data'
_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKSPACE_ROOT = os.path.abspath(
    os.path.join(_PACKAGE_DIR, "..")
)  # Assumes package dir is one level down from root
DEFAULT_INPUT_DIR = os.path.join(_WORKSPACE_ROOT, "input-data/default")

DEFAULT_POINTS_FILE = os.path.join(DEFAULT_INPUT_DIR, "default_points.csv")
DEFAULT_ORIENTATIONS_FILE = os.path.join(DEFAULT_INPUT_DIR, "default_orientations.csv")
DEFAULT_STRUCTURE_FILE = os.path.join(DEFAULT_INPUT_DIR, "default_structure.csv")


def read_file_content(filepath: str) -> str | None:
    """Reads the entire content of a file into a string."""
    if not os.path.isabs(filepath):
        # If path is not absolute, assume it's relative to the workspace root
        filepath = os.path.join(_WORKSPACE_ROOT, filepath)

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
    print("Critical Error: Failed to load essential default data files.")
    print(f"Looked in: {DEFAULT_INPUT_DIR}")
    sys.exit(1)
