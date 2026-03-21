import argparse
from lib.search_utils import get_normalize_scores
from lib.hybrid_search import weighted_command, rrf_search_command

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    normalize_parser = subparsers.add_parser(
    "normalize", help="Normalize scores"
    )
    normalize_parser.add_argument("scores", type=float, nargs='+', help="Scores to be normalized")
    
    weighted_search_parser = subparsers.add_parser(
        "weighted-search", help="Get a weighted search"
    )
    weighted_search_parser.add_argument("query", type=str, help="Search query")
    weighted_search_parser.add_argument("--alpha", type=float, default=0.5, help="Specify weighing of scores")
    weighted_search_parser.add_argument("--limit", type=int, default=5, help="Specify number of search results")
    
    rrf_search_parser = subparsers.add_parser(
        "rrf-search", help="Get a weighted search"
    )
    rrf_search_parser.add_argument("query", type=str, help="Search query")
    rrf_search_parser.add_argument("--k", type=float, default=60, help="Specify weighing of scores")
    rrf_search_parser.add_argument("--limit", type=int, default=5, help="Specify number of search results")
    
    args = parser.parse_args()
    
    match args.command:
        case "normalize":
            result = get_normalize_scores(args.scores)
            for score in result:
                print(f"* {score:.4f}")
        case "weighted-search":
            results = weighted_command(args.query, args.alpha, args.limit)         
            for i, (idx, result) in enumerate(results.items()):
                print(f"{i+1}. {result['title']}")
                print(f"Hybrid Score: {result['hybrid_score']:.4f}")
                print(f"BM25: {result['bm25_score']:.4f}, Semantic: {result["semantic_score"]:.4f}")
                print(f"{result['doc'][:100]}...")
        case "rrf-search":
            results = rrf_search_command(args.query, args.k, args.limit)
            for i, (idx, result) in enumerate(results.items()):
                print(f"{i+1}. {result['document']['title']}")
                print(f"RRF Score: {result['rrf_score']:.4f}")
                print(f"BM25 Rank: {result['bm25_rank']}, Semantic Rank: {result["semantic_rank"]}")
                print(f"{result['document']['description'][:100]}...")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()