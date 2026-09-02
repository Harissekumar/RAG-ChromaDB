from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

# -------------------------
# Environment
# -------------------------

load_dotenv()

print("API Key loaded:", bool(os.getenv("GOOGLE_API_KEY")))

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0
)

response = llm.invoke("What is RAG in AI?")

print("\nGemini Response:")
print(response.content)


# -------------------------
# Step 1: Load Documents
# -------------------------

DATA_PATH = "data/new_articles"

loader = DirectoryLoader(
    DATA_PATH,
    glob="**/*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"}
)

documents = loader.load()

print(f"Loaded {len(documents)} documents")


# -------------------------
# Step 2: Split Documents
# -------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks")


# -------------------------
# Step 3: Embeddings
# -------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded!")


# -------------------------
# Step 4: Store in ChromaDB
# -------------------------

persist_directory = "db"

vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=persist_directory
)

print("Documents stored in ChromaDB!")


# -------------------------
# Step 5: Retriever
# -------------------------

retriever = vectordb.as_retriever(
    search_kwargs={"k": 3}
)

question = "How much did Microsoft invest in OpenAI?"

docs = retriever.invoke(question)


# -------------------------
# Step 6: Display Results
# -------------------------

print("\nRetrieved Documents:")
print("=" * 60)

for i, doc in enumerate(docs):
    print(f"\n--- Result {i + 1} ---")
    print(doc.page_content)