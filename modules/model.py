import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import bitsandbytes as bnb

class ModelHandler:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True
        )
        # Load the model with 4-bit quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct",
            device_map={"": 0},
            quantization_config=quantization_config
        )

        if self.tokenizer.eos_token is None:
            self.tokenizer.add_special_tokens({'eos_token': '<|endoftext|>'})
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

