from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os

from embedder import embed_and_store
load_dotenv()

# LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

# Vectorstore
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

# vectorstore = Chroma(
#     persist_directory="./chroma_db",
#     embedding_function=embeddings
# )


# Step 1 — Load PDF
loader = PyPDFLoader("sample_pdf_adobe.pdf")
documents = loader.load()
print(f"Loaded {len(documents)} pages")

# Step 2 — Chunk
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)
chunks = [c for c in chunks if c.page_content.strip()]
print(f"Split into {len(chunks)} chunks")

# Step 3 — Embed and store
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

vectorstore = Chroma.from_documents(chunks, embeddings)

# Tool 1 — search document
@tool
def search_document(query: str) -> str:
    """Search the document for relevant information based on a query."""
    docs = vectorstore.similarity_search(query, k=3)
    return "\n\n".join(doc.page_content for doc in docs)


# Tool 2 — calculate
@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression. Example: '11.00 + 11.00 + 11.00'"""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


# Create agent
tools = [search_document, calculate]
agent = create_react_agent(llm, tools)


def run_agent(question: str):
    print(f"\n🤖 Question: {question}\n")

    response = agent.invoke({
        "messages": [HumanMessage(content=question)]
    })

    # Print reasoning steps
    for message in response["messages"]:
        print(f"[{message.type}]: {message.content}\n")


if __name__ == "__main__":
    run_agent(
        "What are the transaction amounts in the document and what is their total?"
    )