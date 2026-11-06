# ARC-AGI Solver

A lightweight, research-oriented project for solving Abstract Reasoning Challenge (ARC) tasks using a fine-tuned language model with QLoRA adaptation and a validation sandbox.

This repository combines:

- a pretrained Qwen-based reasoning model
- LoRA adapter weights for ARC-style pattern completion
- grid parsing and ARC-task validation utilities
- a Streamlit frontend for exploring training/test examples and predictions

## Overview

ARC tasks require identifying a transformation from a small set of example input/output grids and applying it to a new test input. This project focuses on the inference and evaluation workflow:

1. load a language model with the trained adapter
2. serialize ARC grids into a model-friendly prompt
3. parse the generated output back into a valid grid
4. validate the result against known training pairs inside a sandbox
5. present tasks and predictions in a browser UI

## Features

- ARC grid serialization and parsing utilities
- Sandbox-based validation for generated Python solver logic
- QLoRA model loading with `transformers` and `peft`
- Streamlit dashboard for interactive task exploration
- Support for uploaded ARC challenge JSON files
- Prediction export in JSON format

## Project Structure

```text
arc_agi_solver/
├── configs/                 
├── frontend/
│   └── app.py               
├── notebook/                
├── src/
│   ├── augmentations.py     
│   ├── model_loader.py      
│   ├── sandbox.py           
│   └── utils.py             
├── tests/
│   └── test_utils.py        
├── weights/
│   └── arc_lora_weights/    
├── LICENSE
├── predictions.json         
├── requirements.txt          
├── .gitignore
└── README.md
```
  
## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/hasna-c/ARC-AGI-Solver.git
cd ARC-AGI-Solver
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Ensure model weights are available

The project expects a LoRA adapter under:

```text
./weights/final_arc_adapter
```

If the adapter is not already present, update the path in `src/model_loader.py` or provide the trained weights in the expected location.

## Running the App

Start the Streamlit interface:

```bash
streamlit run frontend/app.py
```

Then:

- open the local URL shown in the terminal
- upload an ARC challenge JSON file from the sidebar
- inspect the training pairs and test cases
- review the model prediction and exported JSON result

## Example Workflow

1. Load an ARC task JSON containing `train` and `test` examples.
2. Select a task ID from the sidebar.
3. Review the example transformation pairs.
4. Inspect the predicted output for the selected test input.
5. Download the generated grid as JSON for further analysis.

## Model and Inference Notes

The base model is configured as:

```python
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
```

The loader uses 4-bit quantization via `BitsAndBytesConfig` and attaches the trained LoRA adapter with `PeftModel.from_pretrained(...)`.

## Validation and Testing

This project includes a small utility test suite for grid parsing and validation:

```bash
pytest
```

The tests cover:

- grid serialization
- output extraction from model text
- invalid grid rejection
- basic ARC matrix validation

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


## Notes

This repository is designed for experimentation and research workflows. For production-grade deployment or larger-scale evaluation, you may want to add:

- a proper configuration manager
- automated dataset loaders
- a training pipeline for adapter updates
- CI checks and packaging metadata
- improved error handling for model outputs


