import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

class EmbeddingHandler:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = AutoModel.from_pretrained("distilbert-base-uncased")
        self.index = faiss.IndexFlatL2(768)

    def get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.squeeze().numpy()

    def insert_data_into_faiss(self, data):
        vectors = [self.get_embedding(d["documents"]) for d in tqdm(data)]
        #faiss의 index를 맞추기 위해 편의상 첫번째 row만 삽입
        padded_vectors = []
        for vec in vectors:
            padded_vectors.append(vec[0])

        vectors = np.array(padded_vectors)
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        faiss.write_index(self.index, "faiss_index.bin")
