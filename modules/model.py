import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class ModelHandler:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
        self.model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B")
        
        if self.tokenizer.eos_token is None:
            self.tokenizer.add_special_tokens({'eos_token': '<|endoftext|>'})
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
