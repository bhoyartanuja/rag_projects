import chromadb
from fastembed import TextEmbedding
from loader import load_and_chunk
from rag_project.loader import load_and_chunk_with_metadata

# Load embedding model (downloads once, ~80MB)
model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Initialize ChromaDB (local, no server needed)
client = chromadb.PersistentClient(path="./chroma_db")
client.delete_collection(name="rag_collection")
collection = client.get_or_create_collection(name="rag_collection")

def embed_and_store(pdf_path):
    chunks = load_and_chunk_with_metadata(pdf_path)
    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    print(f"Embedding {len(chunks)} chunks...")
    
    embeddings = list(model.embed(texts))
    embeddings = [e.tolist() for e in embeddings]
    
    # Store in ChromaDB
    collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    
    print(f"Stored {len(chunks)} chunks with metadata in ChromaDB")

if __name__ == "__main__":
    embed_and_store("sample_pdf_adobe.pdf")
