import os

from dotenv import load_dotenv
from google import genai
import chromadb
from fastembed import TextEmbedding

load_dotenv()

client_llm = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Load embedding model
model_embed = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Connect to ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="rag_collection")


def rag_query(question, top_k=3):
    # Step 1 - Retrieve relevant chunks
    question_embedding = list(model_embed.embed([question]))[0].tolist()
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k
    )
    chunks = results['documents'][0]
    context = "\n\n".join(chunks)

    # Step 2 - Generate answer using Gemini
    prompt = f"""You are a helpful assistant. 
Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {question}

Answer:"""

    response = client_llm.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    print(f"\n🔍 Question: {question}")
    print(f"\n🤖 Answer: {response.text}")


if __name__ == "__main__":
    rag_query("what is the date in the document?")