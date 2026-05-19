# PDF RAG Pipeline 🔍

A Retrieval Augmented Generation (RAG) pipeline built from scratch — no LangChain, no magic. Every layer is explicit and understandable.

## What This Does
Upload any PDF → ask questions in plain English → get accurate answers grounded in the document.

## Stack
| Tool | Purpose |
|---|---|
| `pypdf` | Extract text from PDF |
| `fastembed` + `BAAI/bge-small-en-v1.5` | Convert text to vectors (ONNX, no PyTorch needed) |
| `ChromaDB` | Local vector store |
| `Google Gemini 2.5 Flash` | Generate natural language answers |
| `python-dotenv` | Manage API keys securely |

## Project Structure
```
rag_project/
├── loader.py         # PDF → text → chunks
├── embedder.py       # chunks → vectors → ChromaDB
├── query.py          # raw similarity retrieval
├── gemini_rag.py     # full RAG — retrieval + Gemini answer
├── requirements.in   # direct dependencies only
├── requirements.txt  # full pinned lockfile
├── .env              # API keys (never committed)
└── .gitignore
```

## Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create `.env`:
```
GEMINI_API_KEY=your_key_here
```

## Run
```bash
# Step 1 — embed your PDF into ChromaDB (run once)
python embedder.py

# Step 2 — ask questions
python gemini_rag.py
```

## How RAG Works
```
SETUP (once):
PDF → extract text → split into chunks → embed chunks → store vectors in ChromaDB

QUERY (every question):
question → embed → find similar vectors → retrieve chunks → send to LLM → answer
```

**Key insight:** The LLM doesn't guess. You hand it the relevant context and it summarises into a clean answer. This grounds the response in your actual data and reduces hallucination.

---

## 📚 Revision Notes — Chunking Strategies

### 1. Fixed Size Chunking
Split every N characters regardless of content.
- ✅ Simple, predictable
- ❌ Can split mid-sentence
- **Use when:** Plain text, quick prototypes

### 2. Fixed Size with Overlap ← used in this project
Each chunk shares N characters with the next chunk.
- ✅ Preserves context at boundaries
- ❌ Slight storage increase
- **Use when:** Default for most RAG systems
- **Our config:** `chunk_size=500, overlap=50`

### 3. Sentence Based
Split on sentence boundaries using NLP (spaCy, NLTK).
- ✅ Semantically clean
- ❌ Variable chunk sizes
- **Use when:** News articles, conversational text

### 4. Paragraph Based
Split on `\n\n` — natural paragraph breaks.
- ✅ Preserves complete thoughts
- ❌ Paragraphs vary in length
- **Use when:** Essays, blog posts

### 5. Recursive Character Splitting ← LangChain default
Try `\n\n` → `\n` → `.` → ` ` recursively until chunk fits.
- ✅ Respects document structure naturally
- ❌ Slightly complex logic
- **Use when:** Mixed format documents, real world default

### 6. Semantic Chunking
Embed sentences, split where similarity drops significantly.
- ✅ Topically coherent chunks
- ❌ Expensive, slower
- **Use when:** High accuracy requirement

### 7. Document Structure Based
Split by headers, sections, chapters.
- ✅ Preserves full semantic sections
- ❌ Requires structured documents
- **Use when:** Legal docs, contracts, manuals, PDFs with clear sections

---

## 📚 Revision Notes — Vector DB Concepts

### Adding new documents to existing ChromaDB
**Full Rebuild strategy** (used here):
- Drop all vectors → re-embed all old + new docs → re-insert
- ✅ Clean, consistent, no stale/duplicate vectors
- ❌ Slower at scale

**Incremental Upsert strategy:**
- Embed only new docs → upsert into existing collection
- ✅ Fast, cheap
- ❌ Risk of duplicates and stale data over time
- Requires deterministic chunk IDs: `md5(doc_name + chunk_index)`

**Interview answer:** Choose based on doc volume and update frequency. Full rebuild for batch pipelines, incremental for real-time high-volume systems.

### Querying across multiple documents
Vector search is inherently batch-aware. One query searches ALL vectors simultaneously — no document boundaries at retrieval time. Returns top-K most similar chunks regardless of source.

### Source concentration problem
Top-K results may all come from one document.
**Solution:** MMR — Maximal Marginal Relevance — balances relevance with diversity across sources.

### Metadata filtering
Tag chunks at upsert time: `source`, `date`, `doc_type`, `department`.
Filter at query time: *"only search HR policy documents"*.

### Performance at scale
Use ANN — Approximate Nearest Neighbor indexing (HNSW algorithm).
ChromaDB and Pinecone use this internally for fast search at millions of vectors.

### Embedding model change = breaking change
All vectors must be re-embedded. Vector spaces are model-specific — you cannot mix vectors from different models in the same collection.

---

## 📚 Revision Notes — RAG Concepts

### Precision vs Recall
- **Precision** — of everything predicted positive, how many were actually positive? (Are my predictions trustworthy?)
- **Recall** — of everything actually positive, how many did I catch? (Did I miss any?)
- **Cancer detection** → high recall (don't miss real cases)
- **Spam filter** → high precision (don't block real emails)

### Why RAG over fine-tuning
- Fine-tuning is expensive and static — needs retraining when data changes
- RAG is dynamic — update your vector DB, answers update immediately
- RAG is auditable — you can see exactly which chunks were retrieved

### Prompt grounding
Always instruct the LLM: *"Answer using ONLY the context below. If not in context, say I don't know."*
This prevents hallucination and keeps answers grounded in your documents.

---

## Next Steps
- [ ] Add metadata (filename, page number) to each chunk
- [ ] Build a simple Flask API around `gemini_rag.py`
- [ ] Add a CLI interface for querying
- [ ] Test with larger, real-world PDFs
- [ ] Explore LangChain to see how it abstracts this pipeline
