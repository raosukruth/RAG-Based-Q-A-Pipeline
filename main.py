import os
import argparse

# the funciton should read all 31 .rst files and return their text with metadata
def load_docs(docs_path):
    # eg {file_name: file_content}
    docs = {}
    for f in os.listdir(docs_path):
        if f.endswith(".rst"):
            with open(os.path.join(docs_path, f), "r",  encoding="utf-8") as file:
                docs[f] = file.read()
    return docs
    

def create_chunks():
    pass

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

    # check if api key is loaded
    api_key = load_api_key()
    print("API key loaded successfully.")
    print(f"API key loaded (length={len(api_key)})")

    docs = load_docs("project1_export/sourcedocs")
    print(len(docs))
    print(list(docs.keys()))





