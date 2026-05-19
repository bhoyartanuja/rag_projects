from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

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
print("Stored in Chroma")

# Step 4 — LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

# Step 5 — Prompt
prompt = PromptTemplate.from_template("""You are a helpful assistant.
Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {question}

Answer:""")

# Step 6 — Chain using LCEL (LangChain Expression Language)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Step 7 — Query
question = "what is this document about?"
answer = chain.invoke(question)

print(f"\n🔍 Question: {question}")
print(f"\n🤖 Answer: {answer}")