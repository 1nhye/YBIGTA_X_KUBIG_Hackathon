# data_preparation.py
import json
import requests

def download_squad():
    url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json"
    response = requests.get(url)
    with open("train-v2.0.json", "w", encoding='utf-8') as f:
        f.write(response.text)

def load_data():
    with open("train-v2.0.json", "r", encoding='utf-8') as f:
        squad = json.load(f)
    texts = []
    for article in squad['data']:
        for paragraph in article['paragraphs']:
            texts.append(paragraph['context'])
    return list(set(texts))  # 중복 제거

if __name__ == "__main__":
    download_squad()
    texts = load_data()
    print(f"총 {len(texts)}개의 문서를 로드했습니다.")

