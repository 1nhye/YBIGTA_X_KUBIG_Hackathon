from transformers import pipeline
import faiss
from embedding import EmbeddingHandler

class AnswerGenerator:
    def __init__(self, model_handler):
        self.model_handler = model_handler
        self.qa_pipeline = pipeline(
            "text-generation",
            model=model_handler.model,
            tokenizer=model_handler.tokenizer,
            max_new_tokens=50,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            temperature=0.7,
            device=0 if model_handler.device == 'cuda' else -1,
            pad_token_id=model_handler.tokenizer.eos_token_id
        )

    def generate_answer_and_collect_results(self, question, data, top_k=10):
        query_vector = EmbeddingHandler().get_embedding(question).reshape(1, -1)
        faiss.normalize_L2(query_vector)
        index = faiss.read_index("faiss_index.bin")
        distances, indices = index.search(query_vector, k=top_k)
        contexts = [data[int(i)]["document"] for i in indices[0]]
        
        prompt = f"Q: {question}\nContext: {contexts[0]}\nA:"
        inputs = self.model_handler.tokenizer(prompt, truncation=True, max_length=1024, return_tensors="pt")
        inputs = {k: v.to(self.model_handler.device) for k, v in inputs.items()}
        generated_text = self.model_handler.model.generate(**inputs, max_new_tokens=50)
        answer = self.model_handler.tokenizer.decode(generated_text[0], skip_special_tokens=True).split("A:")[-1].strip()

        return {
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": data[0]["answer"]
        }
