from datasets import load_from_disk
from embedding import EmbeddingHandler
from model import ModelHandler
from generation import AnswerGenerator
from evaluation import RAGASEvaluator
import logging
import pandas as pd
from tqdm import tqdm

# 로깅 설정
logging.basicConfig(level=logging.INFO)

# 데이터 로드
logging.info("Loading dataset...")
data = load_from_disk("/root/YBIGTA_X_KUBIG_Hackathon/combined_dataset")['context_library']
_question = load_from_disk("/root/YBIGTA_X_KUBIG_Hackathon/combined_dataset")['test']


# # 임베딩 생성 및 삽입
# logging.info("Inserting data into FAISS index...")
# embedding_handler = EmbeddingHandler()
# embedding_handler.insert_data_into_faiss(data)

# 모델 초기화 및 답변 생성
model_handler = ModelHandler()
answer_generator = AnswerGenerator(model_handler)
results = []

# 질문 개수는 200개 이기 떄문에 range(200)을 걸어두었습니다~!
for i in tqdm(range(200)):
    question = _question[i]["question"]
    logging.info(f"Generating answer for question {i + 1}: {question}")
    result = answer_generator.generate_answer_and_collect_results(question, data)
    # Answer를 한번 읽어보세요. 양이 길어 우선은 주석처리해두었습니다!
    logging.info(f"Generated Answer : {result}")
    results.append(result)

# 혹시 잘못 저장될 수 있을 것 같아 디버거를 걸어놓았습니다ㅎㅎ 결과를 한번 확인해보시고 저장하시지요~ 
# 저장 양식은 아래 코드를 따르지 않을 경우 0점처리 됩니다! 
breakpoint()
result = pd.DataFrame(results)
result.to_csv('submission.csv')
# 평가 수행
logging.info("Evaluating results with RAGAS...")
evaluator = RAGASEvaluator()
evaluator.evaluate_with_ragas(results)
logging.info("Evaluation completed successfully.")
