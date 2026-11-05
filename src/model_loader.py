import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_PATH = "./weights/final_arc_adapter"

def load_trained_model():
    """Loads base model in 4-bit and attaches your downloaded Kaggle QLoRA weights."""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model in 4-bit...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        device_map="auto"
    )

    print(f"Attaching fine-tuned LoRA weights from {ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    
    return model, tokenizer

if __name__ == "__main__":
    model, tokenizer = load_trained_model()
    print("Model and adapters loaded successfully in VS Code workspace!")