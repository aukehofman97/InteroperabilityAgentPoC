import json
import pickle
import faiss
import os
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from openai import OpenAI

# -----------------------------
# Load environment variables
# -----------------------------

os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -----------------------------
# Load input JSON data
# -----------------------------
def load_json_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

data = load_json_file("anonymized_data.json")
print(f"✅ Loaded {len(data)} records from JSON.")

# -----------------------------
# Load ontology vector index
# -----------------------------
def load_ontology_index(index_path="ontology.index", chunks_path="ontology_chunks.pkl"):
    index = faiss.read_index(index_path)
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return index, chunks, model

index, chunks, model = load_ontology_index()
print("✅ Ontology index and chunks loaded.")

# -----------------------------
# Retrieve ontology context from FAISS
# -----------------------------
def retrieve_ontology_context(query_text, index, chunks, model, k=5):
    query_embedding = model.encode([query_text])
    D, I = index.search(query_embedding, k)
    return [chunks[i] for i in I[0]]

# -----------------------------
# Load transformation prompt from external file
# -----------------------------
def load_prompt_template(path="prompt.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

prompt_template = load_prompt_template()

# -----------------------------
# Map single key to ontology term via LLM
# -----------------------------
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

# -----------------------------
# Map all keys in first record
# -----------------------------
sample_record = data[0]
new_key_map = {}

print("\n🔁 Mapping keys to ontology terms...")
for key in sample_record.keys():
    print(f"\n🔍 Mapping key: {key}")
    mapped = map_key_to_ontology(key, index, chunks, model, prompt_template)
    print(f"✅ {key} → {mapped}")
    new_key_map[key] = mapped

# -----------------------------
# Replace all keys in full dataset
# -----------------------------
def remap_keys(data, key_map):
    remapped = []
    for record in data:
        new_record = {key_map.get(k, k): v for k, v in record.items()}
        remapped.append(new_record)
    return remapped

remapped_data = remap_keys(data, new_key_map)

# -----------------------------
# Save transformed data to new JSON
# -----------------------------
with open("mapped_data.json", "w", encoding="utf-8") as f:
    json.dump(remapped_data, f, indent=2)
print("\n✅ Transformed data saved to mapped_data.json.")