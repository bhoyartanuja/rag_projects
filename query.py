import chromadb
from fastembed import TextEmbedding

# Load same model used for embedding
model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Connect to existing ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="rag_collection")


def query(question, top_k=3):
    # Embed the question
    question_embedding = list(model.embed([question]))[0].tolist()

    # Search ChromaDB
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k
    )

    print(f"\n🔍 Query: {question}")
    print(f"\n📄 Top {top_k} relevant chunks:\n")

    for i, doc in enumerate(results['documents'][0]):
        print(f"--- Chunk {i + 1} ---")
        print(doc)
        print()


if __name__ == "__main__":
    query("what is this document about?")