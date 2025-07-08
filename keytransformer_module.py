
import json
import pickle
import faiss
import os
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from openai import OpenAI

os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def load_json_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def load_ontology_index(index_path="ontology.index", chunks_path="ontology_chunks.pkl"):
    index = faiss.read_index(index_path)
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return index, chunks, model

def retrieve_ontology_context(query_text, index, chunks, model, k=5):
    query_embedding = model.encode([query_text])
    D, I = index.search(query_embedding, k)
    return [chunks[i] for i in I[0]]

def load_prompt_template(path="prompt.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def map_key_to_ontology(key, index, chunks, model, prompt_template):
    context_chunks = retrieve_ontology_context(key, index, chunks, model, k=5)
    context_text = "\n---\n".join(context_chunks)
    prompt = prompt_template.format(ontology_context=context_text, input_key=key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()

def map_all_keys_to_ontology(data_path="anonymized_data.json", prompt_path="prompt.txt"):
    data = load_json_file(data_path)
    index, chunks, model = load_ontology_index()
    prompt_template = load_prompt_template(prompt_path)
    sample_record = data[0]
    key_map = {}
    for key in sample_record.keys():
        print(f"🔍 Mapping key: {key}")
        mapped = map_key_to_ontology(key, index, chunks, model, prompt_template)
        print(f"✅ {key} → {mapped}")
        key_map[key] = mapped
    remapped_data = remap_keys(data, key_map)
    return remapped_data

def remap_keys(data, key_map):
    return [{key_map.get(k, k): v for k, v in record.items()} for record in data]
