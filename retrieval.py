# retrieval.py
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

class Retriever:
    def __init__(self, index_path="faiss_index.index", texts_path="texts.pkl", model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.index = faiss.read_index(index_path)
        with open(texts_path, "rb") as f:
            self.texts = pickle.load(f)

    def retrieve(self, query, top_k=5):
        query_vec = self.model.encode([query], convert_to_numpy=True)
        D, I = self.index.search(query_vec, top_k)
        results = [self.texts[idx] for idx in I[0]]
        return results

if __name__ == "__main__":
    retriever = Retriever()
    query = "What is the capital of France?"
    results = retriever.retrieve(query)
    for res in results:
        print(res)
