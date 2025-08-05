from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from retriever import get_similar_docs

# Load local Flan-T5 model
tokenizer = AutoTokenizer.from_pretrained("./models/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("./models/flan-t5-base")
pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=128)
llm = HuggingFacePipeline(pipeline=pipe)

def query_agent(query, index_name):  # ✅ Expect index_name
    docs = get_similar_docs(query, index_name)  # ✅ Pass both args

    if isinstance(docs, str):  # Error string
        return docs

    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"Answer this legal question based only on the following context from {index_name}.pdf:\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    response = llm(prompt)
    return response
