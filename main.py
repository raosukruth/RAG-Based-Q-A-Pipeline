import os
import argparse
from langchain_text_splitters import RecursiveCharacterTextSplitter
# import faiss
# from langchain_community.vectorstores import FAISS
# from langchain_openai import OpenAIEmbeddings
# from langchain_core.documents import Document
# from uuid import uuid4
import openai
import json
import pathlib
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
# from langchain_community.retrievers import BM25Retriever
# from langchain.retrievers import EnsembleRetriever


PINECONE_API_KEY = Path("~/pinecone-api-key.txt").expanduser().read_text(encoding="utf-8").splitlines()[0].strip()
PINECONE_INDEX  = "project1-rag-index"
PINECONE_NS     = "chunk1000"
K = 5

def load_docs(docs_path):
    docs = {}
    for f in sorted(os.listdir(docs_path)):
        if f.endswith(".rst"):
            with open(os.path.join(docs_path, f), "r", encoding="utf-8") as fh:
                docs[f] = fh.read()
    return docs


def create_chunks(docs, chunk_size=1000, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
        add_start_index=True,
    )
    documents = []
    for fname, content in docs.items():
        chunks = splitter.create_documents([content], metadatas=[{"source": fname}])
        for i, chunk in enumerate(chunks):
            start = chunk.metadata["start_index"]
            line_start = content[:start].count("\n") + 1
            line_end   = line_start + chunk.page_content.count("\n")
            chunk.metadata.update({
                "chunk_index": i,
                "line_start": line_start,
                "line_end": line_end,
            })
            documents.append(chunk)
    return documents


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

def resolve_index_name(pc, preferred):
    available = [idx.name for idx in pc.list_indexes()]
    print(f"Available Pinecone indexes: {available}")
    if preferred in available:
        return preferred
    if available:
        print(f"Index '{preferred}' not found — using '{available[0]}' instead.")
        return available[0]

def build_pinecone_retriever(k=K):
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5", model_kwargs={"device": "cpu"},
                                        encode_kwargs={"normalize_embeddings": True})
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = resolve_index_name(pc, PINECONE_INDEX)
    index = pc.Index(index_name)
    vector_store = PineconeVectorStore(index=index, embedding=embeddings, namespace=PINECONE_NS)
    base_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
    cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-4-v2", model_kwargs={"device": "cpu"})
    reranker = CrossEncoderReranker(model=cross_encoder, top_n=k)
    return ContextualCompressionRetriever(base_compressor=reranker, base_retriever=base_retriever)


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
            Answer thoroughly and cite specific details from the context.
            Context: {context}
            Question: {query}
            Answer:"""
        }], 
    )
    return response.choices[0].message.content.strip(), context

def load_validation_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data

if __name__ == "__main__":
    
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--input", required=False, help="Path to input validation JSON")
    arg_parser.add_argument("--output", required=False, help="Path to output JSON")
    arg_parser.add_argument("--upload", action="store_true", help="Upload docs to Pinecone instead of running queries")
    arg_parser.add_argument("--docs", default="project1_export/sourcedocs", help="Path to .rst source docs (used with --upload)")
    arg_parser.add_argument("--namespace", default="chunk1000", help="Pinecone namespace (used with --upload)")
    arg_parser.add_argument("--chunk-size", default=1000, type=int, help="Chunk size (used with --upload)")
    arg_parser.add_argument("--overlap", default=150, type=int, help="Chunk overlap (used with --upload)")

    args = arg_parser.parse_args()

    if args.upload:
        print(f"Loading docs from: {args.docs}")
        docs = load_docs(args.docs)
        print(f"  {len(docs)} files loaded")
        print(f"Chunking (size={args.chunk_size}, overlap={args.overlap}) ...")
        documents = create_chunks(docs, chunk_size=args.chunk_size, chunk_overlap=args.overlap)
        print(f"  {len(documents)} chunks created")
        print("Loading embeddings model ...")
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        print(f"Uploading to Pinecone index='{PINECONE_INDEX}' namespace='{args.namespace}' ...")
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX)
        vector_store = PineconeVectorStore(index=index, embedding=embeddings, namespace=args.namespace)
        vector_store.add_documents(documents)
        print("Done.")
        stats = index.describe_index_stats()
        print(f"Index stats: {stats}")
        exit(0)

    if not args.input or not args.output:
        arg_parser.error("--input and --output are required when not using --upload")

    api_key = load_api_key()

    retriever = build_pinecone_retriever(k=K)


    results = []
    validation_data = load_validation_json(args.input)
    for i in validation_data:
        question = i["question"]
        question_id = i["question_id"]
        print(f"Processing question {question_id}: {question}")    

        docs = retriever.invoke(question)   

        print("\n" + "="*80)
        print(f"Q{question_id}: {question}")
        print(f"GT evidence: {i.get('source_evidence', 'N/A')}")
        print("Top-5 retrieved:")
        for d in docs:
            md = d.metadata
            print(f"  file={md['source']!r}  lines=[{md['line_start']},{md['line_end']}]")
            print(f"     text: {d.page_content[:140].replace(chr(10),' ')}")

        response, context = llm_response(question, docs, api_key)
        print(f"Answer for question {question_id}: {response}")

        sources = []
        for doc in docs:
            source = {
                "file": doc.metadata['source'], "lines": [doc.metadata['line_start'], 
                        doc.metadata['line_end']]}
            sources.append(source)

        # response, context = llm_response(question, docs, api_key)
        # sources = [{"file": d.metadata["source"],
        #     "lines": [d.metadata["line_start"], d.metadata["line_end"]]}
        #    for d in docs]

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