# RAG-ChromaDB

RAG-ChromaDB demonstrates the building blocks of a Retrieval-Augmented Generation (RAG) pipeline using LangChain, Hugging Face embeddings, ChromaDB, and Google's Gemini LLM.

## Overview

The project loads a collection of UTF-8 text articles, divides them into manageable chunks, creates local vector embeddings, and stores those embeddings in ChromaDB. It then retrieves documents relevant to a question and connects to Gemini through LangChain for model invocation.

## RAG Architecture

```text
Documents
    -> Document Loading
    -> Text Chunking
    -> Hugging Face Embeddings
    -> ChromaDB Vector Store
    -> Similarity Retrieval
    -> Gemini LLM
    -> Final Answer
```

The current implementation exercises document ingestion, indexing, retrieval, and Gemini invocation. End-to-end context-aware answer generation is part of the roadmap.

## Tech Stack

- Python
- LangChain
- LangChain Community
- LangChain Text Splitters
- Hugging Face / Sentence Transformers
- ChromaDB
- Google Gemini
- python-dotenv

## Current Features

- TXT document ingestion
- UTF-8 document loading
- Recursive text splitting
- Local embedding generation
- 384-dimensional embeddings using `sentence-transformers/all-MiniLM-L6-v2`
- ChromaDB vector storage
- Similarity-based document retrieval
- Gemini LLM integration
- Environment-variable-based API key configuration

## Project Structure

```text
RAG-ChromaDB/
|-- data/
|   `-- new_articles/
|       `-- 21 existing .txt files
|-- src/
|   `-- main.py
|-- db/                 # Local generated ChromaDB data (ignored)
|-- .env                # Local environment variables (ignored)
|-- .gitignore
|-- requirements.txt
`-- README.md
```

## Setup

```powershell
git clone https://github.com/Harissekumar/RAG-ChromaDB.git
cd RAG-ChromaDB

python -m venv .venv
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

Create a local `.env` file in the project root containing:

```text
GOOGLE_API_KEY=your_api_key_here
```

Do not replace this placeholder in the README or commit your real API key.

## Run the Project

```powershell
python src/main.py
```

The first run may download the Hugging Face embedding model and create the local `db/` directory.

## Environment Variables

`GOOGLE_API_KEY` authenticates requests to Google's Gemini API. Keep API keys in the local `.env` file or another secret-management system. Never commit API keys to GitHub.

## Current Progress

The current implementation successfully:

- Loads 21 documents
- Creates 233 chunks
- Generates embeddings
- Stores embeddings in ChromaDB
- Retrieves relevant documents
- Connects to Gemini

## Roadmap

- [x] Document ingestion
- [x] Text chunking
- [x] Hugging Face embeddings
- [x] ChromaDB vector storage
- [x] Similarity retrieval
- [x] Gemini LLM integration
- [ ] Retrieval-Augmented Generation pipeline
- [ ] Context-aware prompting
- [ ] Grounded final answers
- [ ] Source/citation display
- [ ] Better chunking strategy
- [ ] Metadata filtering
- [ ] FastAPI backend
- [ ] React frontend
- [ ] Production-ready RAG application

## Learning Objective

This project is being built to understand RAG and AI Engineering fundamentals through practical implementation.
