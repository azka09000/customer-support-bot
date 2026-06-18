# Modern RAG Chatbot - Semantic Search with SentenceTransformers

A production-ready Retrieval-Augmented Generation (RAG) system that combines semantic search with generative AI for intelligent document-based question answering.

## Architecture

This project implements a **modern RAG pipeline** using semantic embeddings and Gemini LLM:

```
PDF Document
    ↓
[PDF Loader] Extract raw text
    ↓
[Text Chunker] Split into overlapping chunks
    ↓
[SemanticEmbedder] Convert chunks → dense vectors (SentenceTransformers)
    ↓
[Vector Store] Index embeddings with cosine similarity
    ↓
[User Query] → Embed with same model
    ↓
[Similarity Search] Retrieve top-k relevant chunks
    ↓
[Gemini LLM] Generate answer from context
    ↓
Answer
```

## Key Features

- **Semantic Embeddings**: Uses `all-MiniLM-L6-v2` from SentenceTransformers for meaningful vector representations
- **Local Model**: No dependency on expensive embedding APIs (unlike Gemini Embeddings)
- **Fast Retrieval**: Cosine similarity search with NumPy (or FAISS for large-scale)
- **Generative AI**: Gemini LLM for final answer generation based on retrieved context
- **Efficient**: Lightweight embedding model (~80MB) + simple vector operations

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set your Gemini API key
export GEMINI_API_KEY="your_api_key_here"
```

## Usage

```bash
python main.py
```

The system will:
1. Load your PDF from `data/pdfs/academic_regulations.pdf`
2. Split into semantic chunks
3. Compute embeddings using SentenceTransformers
4. Retrieve relevant chunks for the query
5. Generate an answer with Gemini

## Configuration

### Embedding Model

Change the embedding model in `main.py`:

```python
embedder = SemanticEmbedder(model_name="all-MiniLM-L6-v2")
```

Other options:
- `"all-mpnet-base-v2"` - Better quality (slower, larger)
- `"paraphrase-MiniLM-L6-v2"` - Paraphrase-focused
- `"multilingual-e5-small"` - Multilingual support

### Vector Store

FAISS is enabled by default for fast retrieval. Set `USE_FAISS = False` in `main.py` to use NumPy:

```python
USE_FAISS = True  # Fast FAISS indexing (pip install faiss-cpu)
# USE_FAISS = False  # Simple NumPy (always works)
```

### Chunk Size & Overlap

Tune retrieval quality by adjusting parameters in `main.py`:
```python
CHUNK_SIZE = 500      # Characters per chunk
CHUNK_OVERLAP = 100   # Character overlap between chunks
RETRIEVAL_K = 10      # Initial candidates before reranking
```

### Reranking

Results are automatically reranked using a cross-encoder model:
- Initial retrieval: k=10 chunks by semantic similarity
- Reranking: Cross-encoder scores and selects top-3
- Generation: Gemini uses top-3 chunks for answer

This improves answer quality by re-scoring candidates with a task-specific model.

## Project Structure

```
.
├── main.py                      # Entry point with RAG pipeline
├── requirements.txt             # Dependencies
├── data/pdfs/                   # PDF documents
└── src/
    ├── pdf_loader.py           # PDF text extraction
    ├── text_chunker.py         # Text splitting with overlap
    ├── semantic_embedder.py    # SentenceTransformers wrapper
    ├── vector_store.py         # NumPy-based cosine similarity
    ├── faiss_vector_store.py   # FAISS-based fast indexing
    ├── reranker.py             # Cross-encoder reranking
    ├── llm.py                  # Gemini LLM integration
    └── tfidf_embedder.py       # Legacy TF-IDF (not used)
```

## Advantages Over TF-IDF

| Aspect | TF-IDF | Semantic (SentenceTransformers) |
|--------|--------|--------------------------------|
| **Meaning** | Word frequency only | Captures semantic meaning |
| **Synonyms** | Different vectors | Same semantic region |
| **Context** | None | Word relationships |
| **Quality** | Lower retrieval quality | Higher, more relevant results |
| **Speed** | Fast | Fast (with optimized models) |
| **API Cost** | None | None (local model) |

## Performance Notes

- **Model Download**: First run downloads `all-MiniLM-L6-v2` (~80MB)
- **Embedding Time**: ~100ms for 100 chunks
- **Vector Dimension**: 384 (all-MiniLM-L6-v2)
- **Memory**: ~500MB for model + embeddings

## Future Enhancements

- [ ] Add FAISS for indexing large PDFs
- [ ] Implement multi-document retrieval
- [ ] Add chat history/conversation context
- [ ] Deploy with FastAPI web service
- [ ] Add reranking layer (cross-encoder)
- [ ] Implement prompt caching for Gemini
- [ ] Add streaming responses

## License

MIT
