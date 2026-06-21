from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[1]

@dataclass(frozen=True)
class Settings:
    data_dir: Path = ROOT_DIR / "data" / "docs"
    persist_dir: Path = ROOT_DIR / "chroma_db"
    collection_name: str = os.getenv("CHROMA_COLLECTION", "hotel_assistant_collection")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")


def required_open_ai_keys():
    required_keys = ["OPENAI_API_KEY"]
    missing_keys = [key for key in required_keys if not os.getenv(key)]

    if missing_keys:
        raise ValueError("API KEY IS INVALID OR MISSING!")