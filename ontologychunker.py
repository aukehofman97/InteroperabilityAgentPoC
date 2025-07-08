import os
from rdflib import Graph, RDFS
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

def build_ontology_vector_index(ontology_path, faiss_index_path=None, metadata_path=None):
    # Step 1: Load ontology and extract labeled/commented terms
    g = Graph()
    g.parse(ontology_path, format="turtle")
    
    chunks = []
    for s in set(g.subjects()):
        label = g.value(s, RDFS.label)
        comment = g.value(s, RDFS.comment)
        if label or comment:
            text = f"URI: {s}\nLabel: {label}\nComment: {comment}"
            chunks.append(text)

    # Step 2: Embed
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)

    # Step 3: Store in FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    # Optional: Save to disk
    if faiss_index_path and metadata_path:
        faiss.write_index(index, faiss_index_path)
        with open(metadata_path, "wb") as f:
            pickle.dump(chunks, f)

    return index, chunks, model

def retrieve_ontology_context(query_text, index, chunks, model, k=3):
    query_embedding = model.encode([query_text])
    D, I = index.search(np.array(query_embedding), k)
    return [chunks[i] for i in I[0]]

build_ontology_vector_index(
    "CombinedOntologyProfile.ttl",
    faiss_index_path="ontology.index",
    metadata_path="ontology_chunks.pkl"
)