import os
import time
import json
from typing import Optional

from dotenv import load_dotenv
from google import genai
from sentence_transformers import CrossEncoder

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

client = genai.Client(api_key=api_key)
model = "gemma-3-27b-it"

def individual_reranker(query, results, limit) -> str:
    for doc in results:  
        prompt = f"""Rate how well this movie matches the search query.

    Query: "{query}"
    Movie: {doc.get("title", "")} - {doc.get("document", "")}

    Consider:
    - Direct relevance to query
    - User intent (what they're looking for)
    - Content appropriateness

    Rate 0-10 (10 = perfect match).
    Output ONLY the number in your response, no other text or explanation.

    Score:"""

        response = client.models.generate_content(model=model, contents=prompt)
        score = None
        try:
            score = int((response.text or "").strip().strip('"'))
        except:
            print("Output score not parsable")
        #print(doc)
        time.sleep(3)
        doc["rerank_score"] = score
        #results[doc.get("id")]["rerank_score"] = score
    return sorted(results, key=lambda x: x["rerank_score"], reverse=True)[:limit]

def batch_reranker(query, results, limit):
    doc_list_str = str(results)
    prompt = f"""Rank the movies listed below by relevance to the following search query.

Query: "{query}"

Movies:
{doc_list_str}

Return ONLY the movie IDs in order of relevance (best match first). Return a valid JSON list, nothing else.

For example:
[75, 12, 34, 2, 1]

Ranking:"""

    response = client.models.generate_content(model=model, contents=prompt)
    doc_id_ranking = list(map(int, list(json.loads(response.text))[:limit]))
    id_to_doc = {doc["id"]: doc for doc in results}
    final_result = []

    for i, doc_id in enumerate(doc_id_ranking):
        doc = id_to_doc.get(doc_id)
        if doc is not None:
            doc["rerank_rank"] = i + 1
            final_result.append(doc)
    
    return final_result
    
def cross_encoder_reranker(query, results, limit):
    pairs = []
    for doc in results:
        pairs.append([query, f"{doc.get('title', '')} - {doc.get('document', '')}"])
        
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
    # `predict` returns a list of numbers, one for each pair
    scores = cross_encoder.predict(pairs)
    final_results = []
    for i, doc in enumerate(results):
        final_results.append({**doc, "cross_encoder_score":scores[i]})
        
    return sorted(final_results, key= lambda x: x["cross_encoder_score"], reverse=True)[:limit]
    
def rerank_results(query, results, limit, method: Optional[str] = None) -> str:
    match method:
        case "individual":
            return individual_reranker(query, results, limit)
        case "batch":
            return batch_reranker(query, results, limit)
        case "cross_encoder":
            return cross_encoder_reranker(query, results, limit)
        case _:
            return results