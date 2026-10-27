# CSV RAG System

This project is a Retrieval-Augmented Generation (RAG) system designed to dynamically read any CSV file in its directory and answer questions based strictly on that data.

It uses:
- **Local embeddings** (`BAAI/bge-small-en-v1.5` via `sentence-transformers`) — no API quota needed to build the vector store.
- **Google Gemini** (`gemini-flash-latest`) for chat generation.
- **ChromaDB** as the local vector database.
- **LangChain** for orchestration and chaining.

## Project Structure

- `config.py`: Central place for model names, paths, and retrieval settings.
- `ingest.py`: Detects every `.csv` file in the project directory and reads all columns of every row. Rebuilds the local Chroma vector store (`./chroma_db`) automatically whenever a CSV's content hash or the embedding model changes. Can be run standalone: `python ingest.py`.
- `retrieval.py`: Loads (building if needed) the vector store and exposes a retriever.
- `chat.py`: Builds the Gemini prompt/chain and answers a question, returning both the answer text and the source reviews it was grounded in.
- `cli.py`: The interactive REPL loop.
- `main.py`: Thin entry point — just calls `cli.main()`.
- `api.py`: FastAPI web app exposing `/health` and a streaming `POST /chat` (Server-Sent Events), serving the frontend from `static/`.
- `static/index.html`: Minimal single-page chat UI (vanilla JS) that streams tokens and displays source citations as chips.
- `requirements.txt`: Pinned project dependencies.

## Prerequisites

1. **Python**: Ensure you have Python installed.
2. **Gemini API key**: Get a free key from [Google AI Studio](https://aistudio.google.com/apikey) (used for chat generation only).

## Installation

1. Navigate to the project directory:
   ```bash
   cd "RAG stage 1(pizza review)"
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and set your key:
   ```bash
   GOOGLE_API_KEY=your_gemini_api_key_here
   ```

## Usage

1. **Run the interactive application:**
   ```bash
   python main.py
   ```
2. **First Run & Dynamic Updates:**
   On first run (or whenever a CSV is added/modified, or the embedding model changes), the vector store is built automatically from every `.csv` file in the directory. Since embeddings run locally, this is quick and uses no API quota. You can also trigger a rebuild explicitly with `python ingest.py`.
3. **Ask Questions:**
   Once initialized, you will see a prompt: `Ask your question (Enter 'q' to quit):`.
   The system will extract the 3 most relevant rows from your CSVs and output an answer strictly based on them.
   If the context has no relevant information, the model will state that it does not know.
4. Enter `q` to exit the application.

## Web App

Instead of the CLI, run the FastAPI server:
```bash
uvicorn api:app --reload
```
Then open http://127.0.0.1:8000 for the chat UI, or `GET http://127.0.0.1:8000/health` to check the vector store status. `POST /chat` with `{"question": "..."}` streams the answer as Server-Sent Events (a `sources` event first, then `token` events, then `done`).
