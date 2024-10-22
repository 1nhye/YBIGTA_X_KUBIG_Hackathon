# generator.py
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class Generator:
    def __init__(self, model_name='gpt2'):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)

    def generate(self, prompt, max_length=200):
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')
        outputs = self.model.generate(
            inputs, 
            max_new_tokens=100,  # 새로 생성할 최대 토큰 수
            num_beams=5, 
            no_repeat_ngram_size=2, 
            early_stopping=True
        )

        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text

if __name__ == "__main__":
    generator = Generator()
    prompt = "The history of artificial intelligence"
    print(generator.generate(prompt))
