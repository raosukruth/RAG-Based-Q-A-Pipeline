import os
import argparse
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_docs(docs_path):
    docs = {}
    for f in os.listdir(docs_path):
        if f.endswith(".rst"):
            with open(os.path.join(docs_path, f), "r",  encoding="utf-8") as file:
                docs[f] = file.read()
    return docs
    
# chunk by 256  
def create_chunks(docs, chunk_size=256):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, 
                                                    chunk_overlap=50, 
                                                    separators=["\n\n", "\n", ".", " ", ""])
    chunks_dict = {}
    for f, content in docs.items():
        chunks_dict[f] = text_splitter.split_text(content)
    return chunks_dict


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

    # test create_chunks
    chunks_dict = create_chunks(docs)
    print(len(chunks_dict))
    for f, chunks in chunks_dict.items():
        print(f"{f}: {len(chunks)} chunks")





