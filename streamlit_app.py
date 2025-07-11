import streamlit as st
from hutton_lm.pdf_parser import extract_text
from hutton_lm.llm_interface import (
    llm_consolidate_parsed_text,
    llm_generate_dsl_summary,
)
from hutton_lm.model_builder import (
    initialize_geomodel_with_tmp_files,
    load_structural_definitions,
    define_structural_groups,
    compute_model_to_html,
)
from hutton_lm.data_loader import DEFAULT_STRUCTURE_FILE

st.title("Geo-LM Interactive")

uploaded = st.file_uploader("Upload geology PDF", type=["pdf"])

if uploaded and st.button("Submit"):
    with st.spinner("Extracting text"):
        text = extract_text(uploaded)
    if not text:
        st.error("Failed to extract text from file")
        st.stop()
    st.success("Text extracted")
    st.text_area("Excerpt", text[:2000])

    with st.spinner("LLM consolidating"):
        consolidated = llm_consolidate_parsed_text(text)
    st.text_area("Consolidated", consolidated, height=200)

    with st.spinner("Generating DSL"):
        dsl = llm_generate_dsl_summary(consolidated)
    st.text_area("Generated DSL", dsl, height=200)

    with st.spinner("Running GemPy model"):
        model = initialize_geomodel_with_tmp_files("StreamlitModel")
        struct_defs = load_structural_definitions(DEFAULT_STRUCTURE_FILE)
        if struct_defs:
            define_structural_groups(model, struct_defs)
            html_path = compute_model_to_html(model, "model.html")
            with open(html_path) as f:
                html = f.read()
            st.components.v1.html(html, height=600)
        else:
            st.error("Failed to load structural definitions")

st.markdown("## Launch")
st.markdown("Run the app with:")
st.code("streamlit run streamlit_app.py --server.address 0.0.0.0")
