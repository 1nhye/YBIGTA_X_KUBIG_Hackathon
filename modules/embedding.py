import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

class EmbeddingHandler:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = AutoModel.from_pretrained("distilbert-base-uncased").to(self.device)
        self.index = faiss.IndexFlatL2(768)

    def get_embedding(self, text):
        # 입력 텍스트를 토크나이징하고, 입력 텐서를 GPU로 이동
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # 출력 텐서를 CPU로 이동시키고, 넘파이 배열로 변환
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        return embeddings.squeeze()

    def insert_data_into_faiss(self, data):
        vectors = [self.get_embedding(d["documents"]) for d in tqdm(data)]
        
        # faiss의 index를 맞추기 위해 편의상 첫 번째 벡터만 삽입 -> 이러면 성능에 치명적이겠죠? 
        # 1. faiss 대신 다른 VectorDB를 사용하는 방법도 있고, 
        # 2. Embedding 방법, 
        # 3. chunk를 쪼개는 방법 등등 자유롭게 고민해보세요ㅎㅎ
        padded_vectors = []
        for vec in vectors:
            padded_vectors.append(vec[0])

        vectors = np.array(padded_vectors)
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        faiss.write_index(self.index, "faiss_index.bin")
