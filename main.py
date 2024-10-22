from fastapi import FastAPI, Query
from retrieval import Retriever
from llm import Generator
import logging

app = FastAPI()
retriever = Retriever()
generator = Generator()

@app.get("/query")
async def get_answer(question: str = Query(..., description="The question to ask")):
    logging.error(f"Question: {question}")
    retrieved_texts = retriever.retrieve(question)
    print(retrieved_texts)
    context = " ".join(retrieved_texts)
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    answer = generator.generate(prompt)
    return {"answer": answer}
