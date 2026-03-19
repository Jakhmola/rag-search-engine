#!/usr/bin/env python3

import argparse
from lib.semantic_search import (
    SemanticSearch,
    verify_model,
    embed_text,
    verify_embeddings,
    embed_query_text,
    semantic_search_command
)
from lib.search_utils import DEFAULT_SEARCH_LIMIT


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("verify", help="Verify the model")

    embedding_parser = subparsers.add_parser("embed_text", help="Gives text embeddings")
    embedding_parser.add_argument("text", type=str, help="Text to be embedded")

    subparsers.add_parser("verify_embeddings", help="Verify text embeddings")

    query_embedding_parser = subparsers.add_parser(
        "embedquery", help="Embed query text"
    )
    query_embedding_parser.add_argument("query", type=str, help="Text to be embedded")

    semantic_search_parser = subparsers.add_parser(
        "search", help="Gives semantically similar movies"
    )
    semantic_search_parser.add_argument("query", type=str, help="Query for search")
    semantic_search_parser.add_argument(
        "--limit",
        type=int,
        nargs="?",
        default=DEFAULT_SEARCH_LIMIT,
        help="Number of relevant movies to be returned",
    )

    args = parser.parse_args()

    match args.command:
        case "verify":
            ss = SemanticSearch()
            verify_model(ss)
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            embed_query_text(args.query)
        case "search":
            results = semantic_search_command(args.query, args.limit)
            for i, result in enumerate(results):
                print(f"{i}. {result["title"]} ({result["score"]}\n{result["description"]})")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
