from data_loader import DataLoader
from embedding import EmbeddingHandler
from model import ModelHandler
from generation import AnswerGenerator
from evaluation import RAGASEvaluator
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)

# 데이터 로드
logging.info("Loading dataset...")
data_loader = DataLoader("rojagtap/tech-qa")
data = data_loader.load_data()

# # 임베딩 생성 및 삽입
# logging.info("Inserting data into FAISS index...")
# embedding_handler = EmbeddingHandler()
# embedding_handler.insert_data_into_faiss(data)

# 모델 초기화 및 답변 생성
model_handler = ModelHandler()
answer_generator = AnswerGenerator(model_handler)
results = []

for i in range(10):
    question = data[i]["question"]
    logging.info(f"Generating answer for question {i + 1}: {question}")
    result = answer_generator.generate_answer_and_collect_results(question, data)
    results.append(result)

# 평가 수행
logging.info("Evaluating results with RAGAS...")
evaluator = RAGASEvaluator()
evaluator.evaluate_with_ragas(results)
logging.info("Evaluation completed successfully.")
