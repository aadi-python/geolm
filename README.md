# geo-lm

A prototype language model interface for generating geological models\!

![Logo|500](static/image4.png)

## Description

Leveraging the power of the [OpenRouter API](https://openrouter.ai/) and [GemPy](https://www.gempy.org/), `geo-lm` is a Python package designed for generative geology. It utilizes large language models to understand geological information and translate it into 3D geological models.

## Functionality

`geo-lm` employs OpenRouter-compatible language model capabilities to automate the process of creating geological models from existing documentation. The workflow involves the following key steps:

1.  **Document Understanding:** The package can process geology reports and documents by reading OCRed text and interpreting extracted maps.
2.  **Geological Knowledge Consolidation:** The interpreted information for a specific locality is then consolidated into a structured "geology DSL" (Domain Specific Language). This DSL encodes crucial geological knowledge, including:
      * Lithology data (rock types and their properties)
      * Structural interpretations (faults, folds, unconformities)
      * Cross-cutting relations between geological units
      * Relative and absolute time-ordering of geological events.
3.  **3D Model Generation:** Finally, the geology DSL is parsed and used as input for [GemPy](https://www.gempy.org/), an open-source Python library for implicit geological modeling and 3D visualization. This allows for the automatic generation of 3D representations of subsurface geology.

## Examples

![Image](https://github.com/user-attachments/assets/1ad1886b-43a2-44f6-ab92-3c5c3de271aa)

## Installation

`geo-lm` is a Python package managed with Poetry. To set up your environment, ensure you have Poetry installed. If not, you can install it following the instructions on the [official Poetry website](https://python-poetry.org/).

Once Poetry is installed, navigate to the directory containing the `pyproject.toml` file and run:

```bash
poetry install
```

This command will create a virtual environment and install all the necessary dependencies, including GemPy.

## To Run

Before running the example script, set your OpenRouter API key (and optionally a custom base URL):

```bash
export OPENROUTER_API_KEY=<your-key-here>
# export OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

Replace `<your-key-here>` with your actual OpenRouter API key.

The provided example script `run.py` demonstrates how to initiate the geological modeling process using the OpenRouter API. To run it, use the following command:

```bash
python run.py \
    --input-mode llm \
    --prompt-type default \
    --llm-output-dir input-data/llm-generated
```

This command will instruct the script to use the language model (`--input-mode llm`) with a default prompting strategy (`--prompt-type default`) and save the LLM's output (the generated geology DSL) to the `input-data/llm-generated` directory.

### Streamlit interface

An experimental Streamlit app provides a simple web interface for uploading a PDF and viewing the resulting 3‑D model. Make sure the `OPENROUTER_API_KEY` environment variable is set, then run:

```bash
streamlit run streamlit_app.py
```

After the upload completes the model will be displayed interactively in the browser. The app requires `pyvista` with its HTML export extras (`pip install "pyvista[jupyter]"`) for full functionality.

### Notes on Running

This repo was originally tested on the following (open access?) paper:

Patrick B. Redmond, Marco T. Einaudi; The Bingham Canyon Porphyry Cu-Mo-Au Deposit. I. Sequence of Intrusions, Vein Formation, and Sulfide Deposition. Economic Geology 2010;; 105 (1): 43–68. doi: https://doi.org/10.2113/gsecongeo.105.1.43

## Contributing

We welcome contributions to `geo-lm`\! If you have ideas for improvements, new features, or bug fixes, please feel free to open an issue or submit a pull request.

## Notes

This package was originally called `hutton-lm`, but was renamed to `geo-lm` last minute!