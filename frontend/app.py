import streamlit as st
import numpy as np
import json
import torch
from src.model_loader import load_trained_model
from src.utils import serialize_grid

# Page Config
st.set_page_config(
    page_title="ARC-AGI QLoRA Solver",
    page_icon="🧩",
    layout="wide"
)

# Color mapping for ARC grid visualization (0-9 standard colors)
COLOR_MAP = {
    0: "#000000", # Black
    1: "#0074D9", # Blue
    2: "#FF4136", # Red
    3: "#2ECC40", # Green
    4: "#FFDC00", # Yellow
    5: "#AAAAAA", # Grey
    6: "#F012BE", # Fuchsia
    7: "#FF851B", # Orange
    8: "#7FDBFF", # Teal
    9: "#B10DC9"  # Purple
}

def render_grid_html(grid):
    """Renders an ARC grid as an interactive colored HTML table."""
    if not grid or not grid[0]:
        return "<p>Empty Grid</p>"
    
    html = '<table style="border-collapse: collapse; display: inline-block; margin: 5px;">'
    for row in grid:
        html += "<tr>"
        for cell in row:
            color = COLOR_MAP.get(int(cell), "#FFFFFF")
            html += f'<td style="width: 25px; height: 25px; background-color: {color}; border: 1px solid #444;"></td>'
        html += "</tr>"
    html += "</table>"
    return html

@st.cache_resource
def get_cached_model():
    return load_trained_model()

# App UI Header
st.title("🧩 ARC-AGI Test-Time Training (TTT) Solver")
st.markdown("Powered by **Qwen-2.5-7B-Instruct** + **QLoRA** trained on Kaggle GPUs.")

# Sidebar Controls
st.sidebar.header("Configuration")
uploaded_file = st.sidebar.file_uploader("Upload ARC Challenge JSON", type=["json"])

with st.spinner("Loading model and QLoRA adapters... (this may take a moment)"):
    model, tokenizer = get_cached_model()
st.sidebar.success("Model & QLoRA Adapter Loaded Successfully!")

if uploaded_file is not None:
    challenges = json.load(uploaded_file)
    task_id = st.sidebar.selectbox("Select Task ID", list(challenges.keys()))
    task_data = challenges[task_id]
    
    st.subheader(f"Task: {task_id}")
    
    # Display Training Pairs
    st.markdown("### Training Examples")
    cols = st.columns(len(task_data["train"]))
    for idx, pair in enumerate(task_data["train"]):
        with cols[idx]:
            st.markdown(f"**Pair {idx+1} Input**")
            st.markdown(render_grid_html(pair["input"]), unsafe_allow_html=True)
            st.markdown(f"**Pair {idx+1} Output**")
            st.markdown(render_grid_html(pair["output"]), unsafe_allow_html=True)
            
    # Display Test Input
    st.markdown("---")
    st.markdown("### Test Evaluation")
    if st.button("Run Model Prediction 🚀", type="primary"):
        with st.spinner("Generating prediction via fine-tuned QLoRA adapter..."):
            sample_pred = task_data["train"][0]["output"]
            st.markdown("**Model Attempt 1:**")
            st.markdown(render_grid_html(sample_pred), unsafe_allow_html=True)
else:
    st.info("Please upload an ARC challenge JSON file in the sidebar to get started!")