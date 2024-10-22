# build_index.py
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

def embed_texts(texts, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings

def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

if __name__ == "__main__":
    from data_preparation import load_data
    texts = load_data()
    embeddings = embed_texts(texts)
    index = build_faiss_index(embeddings)
    faiss.write_index(index, "faiss_index.index")
    with open("texts.pkl", "wb") as f:
        pickle.dump(texts, f)
    print("FAISS 인덱스와 텍스트를 저장했습니다.")
