from sentence_transformers import SentenceTransformer
import numpy as np
from lib.search_utils import CACHE_DIR, load_movies
import os


class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text: str) -> list[int]:
        if not text.strip():
            raise ValueError("Empty text")
        embedding, *_ = self.model.encode([text.strip()], show_progress_bar=True)
        return embedding

    def build_embedings(self, documents: list[dict]) -> list[int]:
        self.documents = documents
        self.document_map = {}
        doc_list = [f"{doc['title']}: {doc['description']}" for doc in documents]
        self.embeddings = []
        for doc in doc_list:
            self.document_map[doc["id"]] = doc
            self.embeddings.append(self.generate_embedding(doc))
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.save(os.path.join(CACHE_DIR, "movie_embeddings.npy"), self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents: list[dict]):
        self.documents = documents
        self.document_map = {}
        for doc in documents:
            self.document_map[doc["id"]] = doc
        if not os.path.exists(os.path.join(CACHE_DIR, "movie_embeddings.npy")):
            return self.build_embedings(documents)
        self.embeddings = np.load(os.path.join(CACHE_DIR, "movie_embeddings.npy"))
        if len(self.embeddings) == len(self.documents):
            return self.embeddings
        else:
            return self.build_embedings(documents)
    
    def search(self, query, limit):
        if self.embeddings is None or len(self.embeddings) == 0:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        query_embedding = self.generate_embedding(query)
        match_results = []
        for i, doc_embedding in enumerate(self.embeddings):
            match_results.append((cosine_similarity(query_embedding, doc_embedding), self.documents[i]))
        top_n_results = sorted(match_results, key= lambda x: x[0], reverse=True)[:limit]
        return [{"score": result[0], "title": result[1]["title"], "description": result[1]["description"]} for result in top_n_results]
            


def verify_embeddings():
    semantic_search_inst = SemanticSearch()
    documents = load_movies()
    semantic_search_inst.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {semantic_search_inst.embeddings.shape[0]} vectors in {semantic_search_inst.embeddings.shape[1]} dimensions"
    )


def embed_text(text: str) -> list[int]:
    semantic_search_inst = SemanticSearch()
    embedding = semantic_search_inst.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_model(ss: SemanticSearch):
    print(f"Model loaded: {ss.model}")
    print(f"Max sequence length: {ss.model.max_seq_length}")
    

def embed_query_text(query: str):
    semantic_search_inst = SemanticSearch()
    embedding = semantic_search_inst.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def semantic_search_command(query, limit):
    semantic_search_inst = SemanticSearch()
    documents = load_movies()
    semantic_search_inst.load_or_create_embeddings(documents)
    return semantic_search_inst.search(query, limit)
    