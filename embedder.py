import chromadb
from fastembed import TextEmbedding
from loader import load_and_chunk

# Load embedding model (downloads once, ~80MB)
model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Initialize ChromaDB (local, no server needed)
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="rag_collection")

def embed_and_store(pdf_path):
    chunks = load_and_chunk(pdf_path)
    
    print(f"Embedding {len(chunks)} chunks...")
    
    embeddings = list(model.embed(chunks))
    
    # Store in ChromaDB
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    
    print(f"Stored {len(chunks)} chunks in ChromaDB")

if __name__ == "__main__":
    embed_and_store("sample_pdf_adobe.pdf")
