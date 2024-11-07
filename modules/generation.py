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
        # Embed the query question
        query_vector = EmbeddingHandler().get_embedding(question).reshape(1, -1)
        faiss.normalize_L2(query_vector)
        
        # Load FAISS index and retrieve top-k contexts
        index = faiss.read_index("faiss_index.bin")
        distances, indices = index.search(query_vector, k=top_k)
        
        # Collect contexts as a list
        contexts = [data[int(i)]["documents"] for i in indices[0]]
        
        flattened_contexts = [
            item if isinstance(item, str) else " ".join(item) 
            for sublist in contexts for item in (sublist if isinstance(sublist, list) else [sublist])
        ]
        
        # Join contexts for prompt
        _contexts = " ".join(flattened_contexts)
        # Prepare prompt for the model
        prompt = f"Q: {question}\nContext: {_contexts}\nA:"
        inputs = self.model_handler.tokenizer(prompt, truncation=True, max_length=1024, return_tensors="pt")
        inputs = {k: v.to(self.model_handler.device) for k, v in inputs.items()}
        
        # Generate answer
        generated_text = self.model_handler.model.generate(**inputs, max_new_tokens=50)
        answer = self.model_handler.tokenizer.decode(generated_text[0], skip_special_tokens=True).split("A:")[-1].strip()
        
        # Return results with `retrieved_contexts` as a list
        return {
            "question": question,
            "answer": answer,
            "contexts": flattened_contexts,  # Pass as a list for RAGAS
            
            # 직접 eval code를 train set으로 돌려보고 싶다면, train set을 이용해보세요. ground truth를 넣어서 보내야 함! 
            #"ground_truth": data[0]["response"]  # Adjust based on desired ground truth
        }
