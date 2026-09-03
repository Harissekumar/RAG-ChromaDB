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
)

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
    persist_directory=persist_directory,
    collection_name="articles"
)

print("Documents stored in ChromaDB!")


# -------------------------
# Step 5: Create Retriever
# -------------------------

retriever = vectordb.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 10
    }
)


# -------------------------
# Step 6: RAG Question Answering
# -------------------------

question = "How much did Microsoft invest in OpenAI?"


# Retrieve more candidates
docs = retriever.invoke(question)


# -------------------------
# Step 6A: Remove duplicates
# -------------------------

unique_docs = []
seen = set()

for doc in docs:

    content = doc.page_content.strip()

    if content not in seen:
        seen.add(content)
        unique_docs.append(doc)


# -------------------------
# Step 6B: Keyword reranking
# -------------------------

question_words = {
    word.lower().strip(".,?!")
    for word in question.split()
    if len(word) > 2
}


def keyword_score(doc):

    content = doc.page_content.lower()

    score = 0

    for word in question_words:

        if word in content:
            score += 1

    return score


# Sort documents by keyword relevance
unique_docs = sorted(
    unique_docs,
    key=keyword_score,
    reverse=True
)


# Keep the best documents
unique_docs = unique_docs[:5]


# -------------------------
# Step 6C: Build Context
# -------------------------

context = "\n\n--- DOCUMENT ---\n\n".join(
    doc.page_content
    for doc in unique_docs
)


# -------------------------
# Step 7: Create Prompt
# -------------------------

prompt = f"""
You are a question-answering assistant.

Answer the user's question using ONLY the information
contained in the DOCUMENT CONTEXT.

Rules:

1. Use only the provided document context.
2. Carefully look for the exact answer.
3. Pay attention to names, numbers, companies,
   dates and amounts.
4. Do not confuse different investments or amounts.
5. Do not use outside knowledge.
6. Give a short and direct answer.
7. If the answer truly does not exist in the context,
   say exactly:

"I don't know based on the provided documents."

DOCUMENT CONTEXT:
-----------------

{context}

-----------------

USER QUESTION:

{question}

ANSWER:
"""


# -------------------------
# Step 8: Ask Gemini
# -------------------------

response = llm.invoke(prompt)


# -------------------------
# Step 9: Display Results
# -------------------------

print("\n==============================")
print("QUESTION:")
print(question)


print("\n==============================")
print("RETRIEVED DOCUMENTS:")

for i, doc in enumerate(unique_docs, start=1):

    print(f"\n--- Result {i} ---")

    print(doc.page_content)

    print(f"\nKeyword Score: {keyword_score(doc)}")


print("\n==============================")
print("ANSWER:")

print(response.content)