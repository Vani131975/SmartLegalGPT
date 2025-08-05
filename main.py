from fastapi import FastAPI, UploadFile
from agent import query_agent
from retriever import store_docs

app = FastAPI()

@app.post("/upload/")
async def upload_file(file: UploadFile):
    return store_docs(file)

@app.get("/query/")
async def get_answer(q: str):
    return query_agent(q)
