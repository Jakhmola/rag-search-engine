import os

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from .search_utils import get_normalize_scores, load_movies

class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        bm25_results = self._bm25_search(query, 500 * limit)
        semantic_results = self.semantic_search.search_chunks(query, 500 * limit)

        bm25_results_list = []
        bm25_scores_to_norm = []

        for result in bm25_results:
            res = result.split()
            doc_id = int(res[0].strip("()"))
            title = self.idx.docmap[doc_id]["title"]
            description = self.idx.docmap[doc_id]["description"]
            score = float(res[-1])

            bm25_scores_to_norm.append(score)
            bm25_results_list.append({
                "id": doc_id,
                "title": title,
                "description": description,
                "score": score,
            })

        bm25_norm_scores = get_normalize_scores(bm25_scores_to_norm)
        semantic_norm_scores = get_normalize_scores([res["score"] for res in semantic_results])

        hybrid_score_dict = {}

        for result_dict, norm_score in zip(bm25_results_list, bm25_norm_scores):
            hybrid_score_dict[result_dict["id"]] = {
                "doc": result_dict["description"],
                "title": result_dict["title"],
                "bm25_score": norm_score,
                "semantic_score": 0.0,
            }

        for result_dict, norm_score in zip(semantic_results, semantic_norm_scores):
            doc_id = result_dict["id"]

            if doc_id not in hybrid_score_dict:
                hybrid_score_dict[doc_id] = {
                    "doc": result_dict["document"],
                    "title": result_dict["title"],
                    "bm25_score": 0.0,
                    "semantic_score": 0.0,
                }

            hybrid_score_dict[doc_id]["semantic_score"] = norm_score

        # compute hybrid score for ALL docs
        for doc_id, data in hybrid_score_dict.items():
            data["hybrid_score"] = self._hybrid_score(
                data["bm25_score"],
                data["semantic_score"],
                alpha,
            )

        return dict(
            sorted(
                hybrid_score_dict.items(),
                key=lambda item: item[1].get("hybrid_score", 0.0),
                reverse=True,
            )[:limit]
        )
            
            

    def rrf_search(self, query, k=60, limit=5):
        bm25_results = self._bm25_search(query, 500 * limit)
        semantic_results = self.semantic_search.search_chunks(query, 500 * limit)
        
        bm25_map = {}
        for result in bm25_results:
            res = result.split()
            doc_id = int(res[0].strip("()"))
            score = float(res[-1])
            bm25_map.update({doc_id: score})
            
        bm25_doc_ids_ranked = [doc_id for doc_id, _ in sorted(bm25_map.items(), key=lambda x: x[1], reverse=True)] 
        semantic_doc_ids_ranked = [res["id"] for res in sorted(semantic_results, key=lambda d: d["score"], reverse=True)]
        rrf_scores = list(rrf_score(i) for i in range(1, 500 * limit))
        
        rrf_score_dict = {}
        for i, (bm25_id, semantic_id) in enumerate(zip(bm25_doc_ids_ranked, semantic_doc_ids_ranked)):
                        
            if bm25_id not in rrf_score_dict:
                rrf_score_dict.update(
                    {
                        bm25_id: {
                            "document": self.idx.docmap[bm25_id],
                            "bm25_rank": i + 1,
                            "semantic_rank": 0,
                            "rrf_score": rrf_scores[i]}
                        }
                    )
            else:
                rrf_score_dict[bm25_id]["bm25_rank"] = i + 1
                rrf_score_dict[bm25_id]["rrf_score"] += rrf_scores[i]
                
            if semantic_id not in rrf_score_dict:
                rrf_score_dict.update(
                    {
                        semantic_id: {
                            "document": self.idx.docmap[semantic_id],
                            "bm25_rank": 0,
                            "semantic_rank": i + 1,
                            "rrf_score": rrf_scores[i]}
                        }
                    )
            else:
                rrf_score_dict[semantic_id]["semantic_rank"] = i + 1
                rrf_score_dict[semantic_id]["rrf_score"] += rrf_scores[i]
            
        return  dict(sorted(rrf_score_dict.items(), key=lambda x : x[1]["rrf_score"], reverse=True)[:limit])
     
        
    def _hybrid_score(self, bm25_score, semantic_score, alpha=0.5):
        return alpha * bm25_score + (1 - alpha) * semantic_score
    
def rrf_score(rank, k=60):
    return 1 / (k + rank)


def weighted_command(query: str, alpha: float, limit: int) -> dict[dict]:
    documents = load_movies()
    hybrid_searcher = HybridSearch(documents)
    return hybrid_searcher.weighted_search(query, alpha, limit)

def rrf_search_command(query: str, k: int, limit: int) -> dict[dict]:
    documents = load_movies()
    hybrid_searcher = HybridSearch(documents)
    return hybrid_searcher.rrf_search(query, k, limit)