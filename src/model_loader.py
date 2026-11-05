import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_PATH = "./weights/final_arc_adapter"

def load_trained_model():
    print("Loading tokenizer from local cache...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model from local cache in 4-bit...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        device_map="auto",
        local_files_only=True  # <--- Forces it to use already cached files instantly
    )

    print(f"Attaching fine-tuned LoRA weights from {ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    
    return model, tokenizer

if __name__ == "__main__":
    model, tokenizer = load_trained_model()
    print("Model and adapters loaded successfully from local cache!")