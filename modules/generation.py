from transformers import pipeline
import faiss
from embedding import EmbeddingHandler
import numpy as np
import pickle
import re

# class AnswerGenerator:
#     def __init__(self, model_handler):
#         self.model_handler = model_handler
#         self.embedding_handler = EmbeddingHandler()
#         self.qa_pipeline = pipeline(
#             "text-generation",
#             model=model_handler.model,
#             tokenizer=model_handler.tokenizer,
#             max_new_tokens=50,
#             do_sample=True,
#             top_k=5,
#             top_p=0.9,
#             temperature=0.5,
#             device=0 if model_handler.device == 'cuda' else -1,
#             pad_token_id=model_handler.tokenizer.eos_token_id
#         )

#     def generate_answer_and_collect_results(self, question, data, top_k=5, idx=0):
#         # Embed the query question
#         query_vector = self.embedding_handler.get_embedding(question).reshape(1, -1)
#         faiss.normalize_L2(query_vector)

#         # Load FAISS index and retrieve top-k contexts
#         index = faiss.read_index("med_faiss_index.bin")
#         distances, indices = index.search(query_vector, k=top_k)  # 모든 데이터에 대한 거리 계산
        
#         with open("document_references.pkl", "rb") as f:
#             document_references = pickle.load(f)

#         # Collect contexts as a list
#         contexts = [document_references[int(i)] for i in indices[0]]
#         contexts = contexts[:5]

#         ground_truths = data[idx]['response']
    
        
#         # Join contexts for prompt
#         _contexts = " ".join(contexts)

#         # Prepare prompt for the model
#         prompt = f"Q: {question}\nContext: {_contexts}\n A:"
#         inputs = self.model_handler.tokenizer(prompt, truncation=True, max_length=512, return_tensors="pt")
#         inputs = {k: v.to(self.model_handler.device) for k, v in inputs.items()}

#         # Generate answer
#         generated_text = self.model_handler.model.generate(**inputs, max_new_tokens=300)
#         generated_text_decoded = self.model_handler.tokenizer.decode(generated_text[0], skip_special_tokens=True)
#         # print("Generated Text:", generated_text_decoded)
#         answer_parts = re.split(r'\s*A:\s*', generated_text_decoded)
#         answer = answer_parts[-1].strip() if len(answer_parts) > 1 else generated_text_decoded.strip()

#         print(f'answer: {answer}')

#         # Return results with `retrieved_contexts` as a list
#         return {
#             "question": question,
#             "answer": answer,
#             "contexts": contexts,  # Pass as a list for RAGAS
#             "ground_truth": ground_truths  # Adjust based on desired ground truth
#         }

from transformers import pipeline
import faiss
from embedding import EmbeddingHandler
from pymongo import MongoClient
import os
import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle
from keybert import KeyBERT

class AnswerGenerator:
    def __init__(self, model_handler):
        self.model_handler = model_handler
        self.qa_pipeline = pipeline(
            "text-generation",
            model=model_handler.model,
            tokenizer=model_handler.tokenizer,
            max_new_tokens=100,
            do_sample=True,
            top_k=5,
            top_p=0.9,
            temperature=0.3,
            device=0 if model_handler.device == 'cuda' else -1,
            pad_token_id=model_handler.tokenizer.eos_token_id
        )

    def generate_answer_and_collect_results(self, question, data, top_k=5, idx = 0):
        uri = os.getenv("MONGO_URI", "mongodb+srv://cofla2020:dnfl2014!@cluster0.e9ecc.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
        client = MongoClient(uri)
        db = client["hack"]
        collection = db["embedding"]
        
        model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

        kw_model = KeyBERT()

        # 키워드 추출 함수 정의
        def extract_keywords_with_keybert(text, top_n=3):
            # 키워드 추출
            keywords = kw_model.extract_keywords(text, top_n=top_n, keyphrase_ngram_range=(1, 2), stop_words='english')
            # 추출된 키워드만 리스트로 변환
            keywords = [keyword[0] for keyword in keywords]
            return keywords

        # 키워드 기반 문서 검색 함수 정의
        def find_documents_by_keywords(keywords, limit=2):
            documents = []
            for keyword in keywords:
                query = {"text": {"$regex": keyword, "$options": "i"}}
                keyword_docs = collection.find(query).limit(limit)
                for doc in keyword_docs:
                    documents.append(doc["text"])
            return documents



        # Define embedding function
        def get_embedding(text):
            print("Generating embedding for query text...")
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
            with torch.no_grad():
                outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy().tolist()
            print("Embedding generated.")
            return embedding

        query_vector = get_embedding(question)

        keywords = extract_keywords_with_keybert(question)

        
        vector_pipeline = [
            {
            "$vectorSearch": {
                "index": "vector_index",  # Replace with your actual index name
                "exact": False,  # Set to True if you want exact matches only
                "limit": 4,  # Number of results to retrieve
                "numCandidates": 100,  # Number of candidates for similarity ranking
                "path": "embedding",  # Field to search in the collection
                "queryVector": query_vector  # The query embedding
                }
            }
        ]

        vector_results = list(collection.aggregate(vector_pipeline))
        vector_documents = [doc["text"] for doc in vector_results]
        keyword_documents = find_documents_by_keywords(keywords)
        
        results = keyword_documents + vector_documents
        flattened_contexts = [result for result in results]
        flattened_contexts = flattened_contexts[:6]
    


        _contexts = " ".join(flattened_contexts)
        # Prepare prompt for the model
        prompt = f"Q: {question}\nContext: {_contexts}\nA:"
        inputs = self.model_handler.tokenizer(prompt, truncation=True, max_length=1024, return_tensors="pt")
        inputs = {k: v.to(self.model_handler.device) for k, v in inputs.items()}
        
        # Generate answer
        generated_text = self.model_handler.model.generate(**inputs, max_new_tokens=100)
        generated_text_decoded = self.model_handler.tokenizer.decode(generated_text[0], skip_special_tokens=True)
        # print("Generated Text:", generated_text_decoded)
        answer_parts = re.split(r'\s*A:\s*', generated_text_decoded)
        answer = answer_parts[-1].strip() if len(answer_parts) > 1 else generated_text_decoded.strip()
        ground_truths = data[idx]['response']

#         print(f'answer: {answer}')

        print(f'answer: {answer}')
        
        # Return results with `retrieved_contexts` as a list
        return {
            "question": question,
            "answer": answer,
            "contexts": flattened_contexts,  # Pass as a list for RAGAS
            
            # 직접 eval code를 train set으로 돌려보고 싶다면, train set을 이용해보세요. ground truth를 넣어서 보내야 함! 
            "ground_truth": ground_truths  # Adjust based on desired ground truth
        }
