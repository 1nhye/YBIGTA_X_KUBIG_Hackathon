import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle

class EmbeddingHandler:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(self.device)
        # self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        # Load model directly
        
        self.model = AutoModelForMaskedLM.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext").to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
        
        self.index = faiss.IndexFlatIP(768)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)

    def get_embedding(self, text):
        # 입력 텍스트를 토크나이징하고, 입력 텐서를 GPU로 이동
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # 출력 텐서를 CPU로 이동시키고, 넘파이 배열로 변환
        # embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()


        return embeddings.squeeze()

    def insert_data_into_faiss(self, data):
        padded_vectors = []
        # 문서 저장을 위한 리스트
        document_references = []  # 원본 문서 텍스트를 저장할 리스트

        # 2차원 배열 처리
        for doc_list in tqdm(data['documents'], desc="Processing documents"):
            for document in doc_list:  # 각 문서 리스트에서 개별 문서 처리
                if not isinstance(document, str) or not document.strip():  # 문자열인지 확인하고 비어있지 않은지 확인
                    continue  # 유효하지 않은 문서인 경우 건너뜁니다.

                # 각 문서에 대해 청크 생성
                chunks = self.text_splitter.split_text(document)  # LangChain을 이용한 청크 생성
                
                # 각 청크에 대해 임베딩 생성
                for chunk in chunks:
                    embedding = self.get_embedding(chunk)  # 각 청크에 대한 임베딩 생성
                    padded_vectors.append(embedding)
                    document_references.append(chunk)  # 원본 청크 텍스트 저장

        vectors = np.array(padded_vectors)
        faiss.normalize_L2(vectors)
        self.index.add(vectors)  # FAISS 인덱스에 추가
        faiss.write_index(self.index, "med_faiss_index.bin")

        # 문서 참조를 별도의 파일로 저장 (선택사항)
        with open("document_references.pkl", "wb") as f:
            pickle.dump(document_references, f)


