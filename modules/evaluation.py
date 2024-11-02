import requests
import json
import logging
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, faithfulness, context_recall

class RAGASEvaluator:
    def __init__(self):
        self.url = "http://ollama_container:11434/generate"  # Ollama API 엔드포인트

    def generate_with_llama(self, text):
        # Ollama LLaMA 3.1 70B 모델을 통해 평가에 필요한 LLM 응답을 생성합니다
        payload = {
            "model": "llama3.1-70b",
            "text": text
        }
        headers = {"Content-Type": "application/json"}

        response = requests.post(self.url, headers=headers, data=json.dumps(payload))
        if response.status_code == 200:
            response_data = response.json()
            return response_data.get("response", "No response")
        else:
            logging.error(f"Failed to generate response: {response.text}")
            return "Error generating response"

    def evaluate_with_ragas(self, results):
        # Ollama로 각 평가를 수행합니다
        for result in results:
            result["answer"] = self.generate_with_llama(result["question"])

        # 평가할 데이터 샘플 생성
        data_samples = {
            'question': [res["question"] for res in results],
            'answer': [res["answer"] for res in results],
            'contexts': [res["contexts"] for res in results],
            'ground_truth': [res["ground_truth"] for res in results]
        }

        # 평가 수행
        score = evaluate(data_samples, metrics=[answer_relevancy, context_precision, faithfulness, context_recall])
        logging.info("Evaluation Score: %s", score)
