import os
import tempfile
import streamlit as st

from hutton_lm.pdf_parser import extract_text_from_pdf
from hutton_lm.llm_interface import (
    llm_consolidate_parsed_text,
    llm_generate_dsl_summary,
    run_llm_generation,
)
from hutton_lm.model_builder import (
    initialize_geomodel_from_files,
    load_structural_definitions,
    define_structural_groups,
    compute_and_plot_model,
)

st.title("Geo-LM Web Interface")

uploaded_pdf = st.file_uploader("Upload a geology PDF", type=["pdf"])

if uploaded_pdf is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_pdf.read())
        tmp_path = tmp.name

    st.write("Extracting text from PDF...")
    text = extract_text_from_pdf(tmp_path)

    st.write("Consolidating text with LLM...")
    consolidated = llm_consolidate_parsed_text(text)

    st.write("Generating DSL with LLM...")
    dsl_output = llm_generate_dsl_summary(consolidated)

    llm_dir = tempfile.mkdtemp(prefix="llm_output_")
    dsl_file = os.path.join(llm_dir, "geo_dsl.txt")
    with open(dsl_file, "w", encoding="utf-8") as f:
        f.write(dsl_output)

    st.write("Running GemPy model generation...")
    # Use run_llm_generation to produce csvs from DSL
    gen_files = run_llm_generation("DSL", 0.7, llm_dir)
    if not gen_files:
        st.error("LLM generation failed")
    else:
        points_file, orientations_file, structure_file = gen_files
        geo_model = initialize_geomodel_from_files(
            "Streamlit_GeoModel", orientations_file, points_file
        )
        structural_defs = load_structural_definitions(structure_file)
        if structural_defs is None:
            st.error("Failed to load structural definitions")
        else:
            define_structural_groups(geo_model, structural_defs)
            html = compute_and_plot_model(geo_model, return_html=True)
            if html:
                st.components.v1.html(html, height=600, width=800)
            else:
                st.error("Failed to generate plot HTML")

st.write(
    "Make sure the environment variable `DEEPSEEK_API_KEY` is set before running this app."
)
