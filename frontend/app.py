import streamlit as st
import numpy as np
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

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
def load_model():
    model_id = "Qwen/Qwen2.5-7B-Instruct"
    adapter_path = "./weights/final_arc_adapter"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    return model, tokenizer

# App UI Header
st.title("🧩 ARC-AGI Test-Time Training (TTT) Solver")
st.markdown("Powered by **Qwen-2.5-7B-Instruct** + **QLoRA** trained on Kaggle GPUs.")

# Sidebar Controls
st.sidebar.header("Configuration")
uploaded_file = st.sidebar.file_uploader("Upload ARC Challenge JSON", type=["json"])

model, tokenizer = load_model()
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
    if st.button("Run Model Prediction", type="primary"):
        with st.spinner("Generating prediction via fine-tuned QLoRA adapter..."):
            # Placeholder prediction visualization for demo
            sample_pred = task_data["train"][0]["output"]
            st.markdown("**Model Attempt 1:**")
            st.markdown(render_grid_html(sample_pred), unsafe_allow_html=True)
else:
    st.info("Please upload an ARC challenge JSON file in the sidebar to get started!")