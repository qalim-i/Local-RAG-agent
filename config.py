from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "chroma_db"
MANIFEST_PATH = DB_DIR / "source_manifest.json"

COLLECTION_NAME = "csv_documents"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CHAT_MODEL = "gemini-flash-latest"
TOP_K = 3
