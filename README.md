# RAG Search Engine

A movie search engine with full Retrieval-Augmented Generation (RAG) capabilities. It searches a movie database using multiple retrieval strategies — keyword (BM25), semantic (dense embeddings), hybrid fusion, and multimodal (image-based) — and augments results with LLM-generated answers via Google Gemini.

The domain is framed around **Hoopla**, a fictional movie streaming service.

---

## Features

- **BM25 keyword search** with TF-IDF scoring and Porter stemming
- **Dense semantic search** using `all-MiniLM-L6-v2` (384-dim embeddings), with fixed-size and semantic chunking
- **Hybrid search** combining BM25 + semantic via weighted fusion or Reciprocal Rank Fusion (RRF)
- **Query enhancement** — spell correction, query rewriting, and query expansion via Gemma LLM
- **Re-ranking** — LLM-based (individual or batch) and CrossEncoder re-ranking
- **Multimodal search** — image-to-movie search using CLIP (`clip-ViT-B-32`)
- **Augmented generation** — RAG pipeline generating answers, citations, summaries, and Q&A via `gemini-2.5-flash`
- **Evaluation** — Precision@k, Recall@k, F1 over a golden dataset, plus LLM-as-judge scoring

---

## Project Structure

```
rag-search-engine/
├── pyproject.toml
├── data/
│   ├── movies.json          # Movie database (id, title, description)
│   ├── golden_dataset.json  # Evaluation test cases (query → relevant movie titles)
│   └── stopwords.txt        # English stopwords for BM25 tokenization
├── cache/                   # Persisted embeddings and index artifacts
│   ├── movie_embeddings.npy
│   ├── chunk_embeddings.npy
│   ├── chunk_metadata.json
│   ├── index.pkl
│   ├── docmap.pkl
│   ├── term_frequencies.pkl
│   └── doc_lengths.pkl
├── cli/
│   ├── keyword_search_cli.py
│   ├── semantic_search_cli.py
│   ├── hybrid_search_cli.py
│   ├── augmented_generation_cli.py
│   ├── evaluation_cli.py
│   ├── multimodal_search_cli.py
│   ├── describe_image_cli.py
│   ├── test_gemini.py
│   └── lib/
│       ├── search_utils.py          # Shared constants, data loaders, result formatter
│       ├── keyword_search.py        # BM25 inverted index
│       ├── semantic_search.py       # Dense embeddings and chunking
│       ├── hybrid_search.py         # BM25 + semantic fusion
│       ├── query_enhancement.py     # LLM-powered query rewriting
│       ├── reranking.py             # LLM and CrossEncoder re-ranking
│       ├── multimodal_search.py     # CLIP image search
│       ├── augmented_generation.py  # RAG generation pipeline
│       └── evaluation.py            # IR metrics and LLM judge
```

---

## Setup

### Requirements

- Python >= 3.14
- A Google Gemini API key

### Install

```bash
pip install -e .
```

Or install dependencies directly:

```bash
pip install google-genai sentence-transformers numpy nltk pillow python-dotenv loguru
```

### Environment

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_api_key_here
HF_TOKEN=your_huggingface_token  # optional, for gated models
```

---

## Usage

All CLIs live under `cli/` and are run with `python cli/<name>.py <subcommand> [options]`.

### Keyword Search (BM25)

```bash
# Build the inverted index
python cli/keyword_search_cli.py build

# Search using BM25
python cli/keyword_search_cli.py bm25search "animated superhero family"

# Inspect scores
python cli/keyword_search_cli.py tf "batman" 42
python cli/keyword_search_cli.py idf "batman"
python cli/keyword_search_cli.py tfidf "batman" 42
python cli/keyword_search_cli.py bm25idf "batman"
python cli/keyword_search_cli.py bm25tf "batman" 42
```

### Semantic Search

```bash
# Build and cache embeddings
python cli/semantic_search_cli.py verify_embeddings

# Dense vector search
python cli/semantic_search_cli.py search "a girl falls into a fantasy world"

# Chunked semantic search
python cli/semantic_search_cli.py embed_chunks
python cli/semantic_search_cli.py search_chunked "robot learns to feel emotions"

# Inspect chunking strategies
python cli/semantic_search_cli.py chunk "some long text here"
python cli/semantic_search_cli.py semantic_chunk "some long text here"
```

### Hybrid Search

```bash
# Weighted fusion (alpha controls BM25 vs semantic balance, 0.0–1.0)
python cli/hybrid_search_cli.py weighted-search "space adventure" --alpha 0.5

# Reciprocal Rank Fusion
python cli/hybrid_search_cli.py rrf-search "cute animated bear"

# With query enhancement and re-ranking
python cli/hybrid_search_cli.py rrf-search "moovie abut a lost robot" \
  --enhance spell \
  --rerank-method cross_encoder \
  --limit 5
```

`--enhance` options: `spell`, `rewrite`, `expand`  
`--rerank-method` options: `individual`, `batch`, `cross_encoder`

### Augmented Generation (RAG)

```bash
# Full RAG answer
python cli/augmented_generation_cli.py rag "What movies are about time travel?"

# With citations
python cli/augmented_generation_cli.py citations "sci-fi films with AI themes"

# Multi-document summary
python cli/augmented_generation_cli.py summarize "animated movies for kids"

# Conversational Q&A
python cli/augmented_generation_cli.py question "What should I watch on a rainy day?"
```

### Multimodal Search (Image → Movies)

```bash
# Search movies using an image
python cli/multimodal_search_cli.py image_search /path/to/image.jpg --limit 5

# Verify CLIP image embedding
python cli/multimodal_search_cli.py verify_image_embedding /path/to/image.jpg
```

### Image-Assisted Query Description

```bash
# Let Gemini rewrite a text query using an image as context
python cli/describe_image_cli.py --image /path/to/image.jpg --query "movies like this"
```

### Evaluation

```bash
# Run Precision@k, Recall@k, F1 over the golden dataset
python cli/evaluation_cli.py --limit 5
```

---

## Architecture

```
Query
  │
  ├─ [optional] Query Enhancement (spell / rewrite / expand via gemma-3-27b-it)
  │
  ├─ BM25 path: tokenize → stem → BM25 score all docs
  │
  ├─ Semantic path: embed query → cosine sim over chunk embeddings → max-pool per movie
  │
  ├─ Fusion: Reciprocal Rank Fusion  OR  α·BM25 + (1-α)·semantic
  │
  ├─ [optional] Re-ranking: LLM batch / LLM individual / CrossEncoder
  │
  └─ [optional] Generation: gemini-2.5-flash → answer / summary / citations / Q&A
```

### Key Design Decisions

- **Overfetching**: retrieval fetches `limit × 5` candidates before re-ranking or slicing to the final result count
- **Lazy cache**: embeddings and the BM25 index are only rebuilt if cache files are absent or counts mismatch
- **Standardized results**: `format_search_result()` enforces a consistent schema (`id`, `title`, `document`, `score`, `metadata`) across all modules
- **`HybridSearch` as central entrypoint**: all higher-level pipelines (RAG, evaluation) compose through `HybridSearch`, which internally owns both `InvertedIndex` and `ChunkedSemanticSearch`
- **Dispatcher pattern**: `enhance_query()` and `rerank()` use Python structural pattern matching (`match`/`case`) to dispatch to strategy implementations

---

## Models Used

| Model | Purpose |
|---|---|
| `all-MiniLM-L6-v2` | Text embeddings for semantic/hybrid search (384-dim) |
| `clip-ViT-B-32` | Shared image+text embeddings for multimodal search |
| `cross-encoder/ms-marco-TinyBERT-L2-v2` | CrossEncoder re-ranking |
| `gemma-3-27b-it` | Query enhancement, LLM re-ranking, evaluation/judging |
| `gemini-2.5-flash` | RAG answer generation |

---

## Data

`data/movies.json` contains movies with `id`, `title`, and a detailed `description` (plot synopsis). The full `title + description` string is used for both BM25 indexing and embedding.

`data/golden_dataset.json` maps natural-language queries to lists of expected relevant movie titles and is used exclusively for evaluation.
