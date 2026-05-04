import os
import argparse
from langchain_text_splitters import RecursiveCharacterTextSplitter
import faiss
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from uuid import uuid4
import openai

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
    
def create_chunks(docs, chunk_size=256):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, 
                                                    chunk_overlap=50, 
                                                    separators=["\n\n", "\n", ".", " ", ""])
    chunks_dict = {}
    for f, content in docs.items():
        chunks_dict[f] = text_splitter.split_text(content)
    return chunks_dict

def modify_chunks_for_faiss(chunks_dict):
    documents = []
    for f, chunks in chunks_dict.items():
        for i, chunk in enumerate(chunks):
            metadata = {"source": f, "chunk_index": i}
            documents.append(Document(page_content=chunk, metadata=metadata))
    return documents

def create_faiss_index(docs, api_key):
    embeddings = OpenAIEmbeddings(model="api-tgpt-embeddings", openai_api_key=api_key, 
                                    openai_api_base="https://tritonai-api.ucsd.edu")
    
    vector_store = FAISS.from_documents(docs, embeddings) 
    uuids = [str(uuid4()) for _ in range(len(docs))]
    vector_store.add_documents(documents=docs, ids=uuids)

    return vector_store


if __name__ == "__main__":
    # arg_parser = argparse.ArgumentParser()
    # arg_parser.add_argument("--input", required=True, help="Path to input file")
    # arg_parser.add_argument("--output", required=True, help="Path to output file")

    # args = arg_parser.parse_args()

    api_key = load_api_key()
    print("API key loaded successfully.")
    print(f"API key loaded (length={len(api_key)})")

    docs = load_docs("project1_export/sourcedocs")
    print(len(docs))
    print(list(docs.keys()))

    chunks_dict = create_chunks(docs)
    print(len(chunks_dict))
    for f, chunks in chunks_dict.items():
        print(f"{f}: {len(chunks)} chunks")

    documents = modify_chunks_for_faiss(chunks_dict)
    vector_store = create_faiss_index(documents, api_key)
    print("FAISS index built successfully")
    print(f"Total documents in index: {vector_store.index.ntotal}")