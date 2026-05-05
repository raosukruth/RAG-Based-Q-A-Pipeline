import os
import argparse
from langchain_text_splitters import RecursiveCharacterTextSplitter
import faiss
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from uuid import uuid4
import openai
import json


def load_api_key():
    api_key = os.getenv("API_KEY")
    if api_key:
        return api_key

    try:
        file_path = os.path.expanduser("~/api-key.txt")

        with open(file_path, "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        raise ValueError("API key was not found in environment or api_key.txt")

def load_docs(docs_path):
    docs = {}
    for f in os.listdir(docs_path):
        if f.endswith(".rst"):
            with open(os.path.join(docs_path, f), "r",  encoding="utf-8") as file:
                docs[f] = file.read()
    return docs
    
def create_chunks(docs, chunk_size=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, 
                                                    chunk_overlap=50, 
                                                    separators=["\n\n", "\n", ".", " ", ""])
    chunks_dict = {}
    for f, content in docs.items():
        chunks_dict[f] = text_splitter.split_text(content)
    return chunks_dict


def modify_chunks_for_faiss(chunks_dict, docs):
    documents = []
    for f, chunks in chunks_dict.items():
        for i, chunk in enumerate(chunks):
            char_pos = docs[f].find(chunk)
            line_start = docs[f][:char_pos].count("\n") + 1
            line_end = line_start + len(chunk.splitlines()) - 1
            metadata = {"source": f, "chunk_index": i,
                        "line_start": line_start, "line_end": line_end}
            documents.append(Document(page_content=chunk, metadata=metadata))
    return documents

def create_faiss_index(docs, api_key):
    embeddings = OpenAIEmbeddings(model="api-tgpt-embeddings", openai_api_key=api_key, 
                                    openai_api_base="https://tritonai-api.ucsd.edu")
    
    vector_store = FAISS.from_documents(docs, embeddings) 
    return vector_store

def similarity_search(vector_store, query, k=10):
    results = vector_store.similarity_search(query, k=k)
    return results

def llm_response(query, docs, api_key):
    context = ""
    for doc in docs:
        context += doc.page_content + "\n"
    llm = openai.OpenAI(api_key=api_key, base_url="https://tritonai-api.ucsd.edu")
    response = llm.chat.completions.create(
        model="claude-sonnet-4-6",
        messages=[
        {
            "role": "user",
            "content": f"""You are a helpful assistant that answers questions based only on the provided context.
                            If the answer is not in the context, say so briefly.
                            Context: {context} 
                            Question: {query} 
                            Answer:"""
        }
    ]
    )
    return response.choices[0].message.content.strip(), context

def load_validation_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data

if __name__ == "__main__":
    
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--input", required=True, help="Path to input file")
    arg_parser.add_argument("--output", required=True, help="Path to output file")

    args = arg_parser.parse_args()

    api_key = load_api_key()
    docs = load_docs("project1_export/sourcedocs")
    chunks_dict = create_chunks(docs)
    documents = modify_chunks_for_faiss(chunks_dict, docs)
    vector_store = create_faiss_index(documents, api_key)

    results = []
    validation_data = load_validation_json(args.input)
    for i in validation_data:
        question = i["question"]
        question_id = i["question_id"]
        print(f"Processing question {question_id}: {question}")    

        docs = similarity_search(vector_store, question, k=5)
        response, context = llm_response(question, docs, api_key)
        print(f"Answer for question {question_id}: {response}")

        sources = []
        for doc in docs:
            source = {
                "file": doc.metadata['source'], "lines": [doc.metadata['line_start'], 
                        doc.metadata['line_end']]}
            sources.append(source)

        result = {"question_id": question_id, "answer": response, 
                    "retrieved_context": context, "sources": sources}
        
        results.append(result)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=4)
             
    
    # api_key = load_api_key()
    # print("API key loaded successfully.")
    # print(f"API key loaded (length={len(api_key)})")

    # docs = load_docs("project1_export/sourcedocs")
    # print(len(docs))
    # print(list(docs.keys()))

    # chunks_dict = create_chunks(docs)
    # print(len(chunks_dict))
    # for f, chunks in chunks_dict.items():
    #     print(f"{f}: {len(chunks)} chunks")

    # documents = modify_chunks_for_faiss(chunks_dict)
    # vector_store = create_faiss_index(documents, api_key)
    # print("FAISS index built successfully")
    # print(f"Total documents in index: {vector_store.index.ntotal}")

    # results = similarity_search(vector_store, "What is RAG?", k=5)
    # for doc in results:
    #     print(doc.metadata['source'])
    #     print(doc.page_content[:100])

    # response = llm_response("What is RAG?", results, api_key)
    # print("LLM response:")
    # print(response)