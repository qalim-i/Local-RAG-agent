# CSV RAG System

This project is a Retrieval-Augmented Generation (RAG) system designed to dynamically read any CSV file in its directory and answer questions based strictly on that data.

It uses:
- **Google Gemini** (via `langchain-google-genai`) for embeddings and chat generation.
- **ChromaDB** as the local vector database.
- **LangChain** for orchestration and chaining.

## Project Structure

- `vectorise.py`: Detects every `.csv` file in the directory and reads all columns of every row dynamically. If any CSV is added, modified, or the embedding model changes, it automatically rebuilds the local Chroma vector database (`./chroma_db`) using the `text-embedding-004` model.
- `main.py`: The interactive application. It retrieves relevant context from ChromaDB and uses the `gemini-2.5-flash` model to generate an answer based purely on the CSV data.
- `requirements.txt`: Project dependencies.

## Prerequisites

1. **Python**: Ensure you have Python installed.
2. **Gemini API key**: Get a free key from [Google AI Studio](https://aistudio.google.com/apikey).

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
   When `python main.py` is run, it executes `vectorise.py` in the background. It will automatically detect any `.csv` files in the directory, read their contents, and cache them in the vector store (`./chroma_db`). If you modify any CSV files, add new ones, or change the embedding model, it detects the change and recreates the database automatically.
3. **Ask Questions:**
   Once initialized, you will see a prompt: `Ask your question (Enter 'q' to quit):`.
   The system will extract the 3 most relevant rows from your CSVs and output an answer strictly based on them.
   If the context has no relevant information, the model will state that it does not know.
4. Enter `q` to exit the application.
