from openai import OpenAI
client = OpenAI(
    api_key=open("/home/jovyan/api-key.txt").read().strip(),
    base_url="https://tritonai-api.ucsd.edu/..."  # confirm from snippet
)

# generation
resp = client.chat.completions.create(
    model="api-mistral-small-3.2-2506",
    messages=[{"role":"system","content":sys},{"role":"user","content":user}],
    max_tokens=400,
)

# embeddings
emb = client.embeddings.create(
    model="api-tgpt-embeddings",
    input=["chunk text 1", "chunk text 2"],
).data

