import sys
import os

# Fix path to allow importing from the parent directory (src/)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import json
from src.utils import deserialize_grid, is_valid_grid

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
    if not is_valid_grid(grid):
        return "<p>Grid could not be displayed.</p>"

    html = '<table style="border-collapse: collapse; display: inline-block; margin: 5px;" aria-label="ARC grid">'
    for row in grid:
        html += "<tr>"
        for cell in row:
            color = COLOR_MAP.get(int(cell), "#FFFFFF")
            html += f'<td title="Color {int(cell)}" style="width: 30px; height: 30px; background-color: {color}; border: 1px solid #777;"></td>'
        html += "</tr>"
    html += "</table>"
    return html

def show_prediction(grid, raw_output):
    if grid:
        st.success(f"Prediction ready · {len(grid)} rows × {len(grid[0])} columns")
        st.markdown("#### Predicted output")
        st.markdown(render_grid_html(grid), unsafe_allow_html=True)
        st.download_button(
            "Download prediction as JSON",
            data=json.dumps(grid, indent=2),
            file_name="arc_prediction.json",
            mime="application/json",
        )
        st.code(json.dumps(grid), language="json")
    else:
        st.warning("The model response did not contain a valid ARC grid. Check the raw response below.")

    # Expander sits cleanly full-width at the bottom of the section
    with st.expander("View raw model response"):
        st.text(raw_output or "No response text was produced.")

# App UI Header
st.title("🧩 ARC-AGI Test-Time Training (TTT) Solver")
st.markdown("Powered by **[Qwen/Qwen2.5-7B-Instruct](https://www.kaggle.com/code/hasnac/arc-agi-test-time-training-qlora/edit#site-content)** + **QLoRA** trained on Kaggle GPUs.")

# Load pre-computed predictions.json if available
predictions = {}
predictions_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../predictions.json'))
if os.path.exists(predictions_path):
    with open(predictions_path, "r") as f:
        predictions = json.load(f)

# Sidebar Controls
st.sidebar.header("Configuration")
uploaded_file = st.sidebar.file_uploader("Upload ARC Challenge JSON", type=["json"])

if uploaded_file is not None:
    try:
        challenges = json.load(uploaded_file)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        st.error(f"Could not read the uploaded JSON file: {error}")
        st.stop()

    if not isinstance(challenges, dict) or not challenges:
        st.error("The uploaded file must contain an ARC challenge object with at least one task.")
        st.stop()

    task_id = st.sidebar.selectbox("Select Task ID", list(challenges.keys()))
    task_data = challenges[task_id]
    
    st.subheader(f"Task: {task_id}")

    # Display Training Pairs
    st.markdown("### Training Examples")
    train_pairs = task_data.get("train", [])
    cols = st.columns(min(len(train_pairs), 4)) if train_pairs else []
    for idx, pair in enumerate(train_pairs[:4]):
        with cols[idx % len(cols)]:
            st.markdown(f"**Pair {idx+1} Input**")
            st.markdown(render_grid_html(pair["input"]), unsafe_allow_html=True)
            st.markdown(f"**Pair {idx+1} Output**")
            st.markdown(render_grid_html(pair["output"]), unsafe_allow_html=True)
            
    # Display Test Evaluation & Predictions
    st.markdown("---")
    st.markdown("### Test Evaluation & Predictions")

    test_cases = task_data.get("test", [])
    if test_cases:
        test_index = st.selectbox(
            "Test case",
            range(len(test_cases)),
            format_func=lambda index: f"Test {index + 1}",
        )
        test_input = test_cases[test_index]["input"]

        input_col, output_col = st.columns(2)
        with input_col:
            st.markdown("#### Test input")
            st.caption(f"{len(test_input)} rows × {len(test_input[0])} columns")
            st.markdown(render_grid_html(test_input), unsafe_allow_html=True)

        with output_col:
            st.markdown("#### Test output prediction")
            if task_id in predictions:
                raw_output = predictions[task_id]
                prediction = deserialize_grid(raw_output)
                st.caption("Loaded from pre-computed predictions.json")
                show_prediction(prediction, raw_output)
            else:
                st.info("No pre-computed prediction found for this specific task ID in predictions.json.")
    else:
        st.info("This task has no test inputs.")

else:
    st.info("Upload your ARC challenge JSON file in the sidebar to browse tasks and inspect your Kaggle predictions.")