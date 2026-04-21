import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss

#This code is a simplified implementation provided to illustrate the overall structure and 
#does not guarantee full or stable functionality.

# =========================
# config
# =========================
TOP_K = 2
MODEL_NAME = "intfloat/e5-base-v2"

# =========================
# load
# =========================
def load_data(path="data/Maindata_question_Hotpot_sample.csv"):
    df = pd.read_csv(path)
    questions = df["question"].tolist()
    documents = df["gold_documents"].tolist()
    return questions, documents

# =========================
# embedding model
# =========================
model = SentenceTransformer(MODEL_NAME)

def embed_texts(texts):
    texts = [f"query: {t}" for t in texts]
    emb = model.encode(texts, normalize_embeddings=True)
    return np.array(emb).astype("float32")

# =========================
# build Index
# =========================
def build_index(questions):
    embeddings = embed_texts(questions)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index, embeddings

# =========================
# QI-RAG Retrieval (simple)
# =========================
def retrieve_qi_rag(query, index, questions, documents, top_k=2):
    """
    NOTE:
    This is a simplified version for demonstration.
    Core retrieval logic is abstracted for patent protection.
    """
    
    q_emb = embed_texts([query])
    
    scores, indices = index.search(q_emb, top_k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            "rank": i + 1,
            "score": float(scores[0][i]),
            "matched_question": questions[idx],
            "matched_document": documents[idx]
        })
    
    return results

# =========================
# Demo
# =========================
def main():
    print("=== QI-RAG Demo ===")
       
    questions, documents = load_data()
        
    index, _ = build_index(questions)
    
    query = "Which magazine started earlier, Arthur's or First?"
    
    print(f"\nQuery: {query}")
    
    results = retrieve_qi_rag(query, index, questions, documents, TOP_K)
    
    print("\n=== Retrieval Results ===")
    for r in results:
        print(f"[Rank {r['rank']}] Score: {r['score']:.4f}")
        print("Q:", r["matched_question"])
        print("Doc:", r["matched_document"][:100], "...\n")

if __name__ == "__main__":
    main()
