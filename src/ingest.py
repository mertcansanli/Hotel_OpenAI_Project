import argparse
import os
from uuid import uuid4 

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import Settings, required_open_ai_keys
from src.document_loader import load_documents

import shutil 

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200, separators=["\n\n","\n"," ",""])
    return text_splitter.split_documents(documents)

def build_index(reset: bool = False) -> dict :
    required_open_ai_keys()
    settings = Settings()
    
    if reset and settings.persist_dir.exists():
        shutil.rmtree(settings.persist_dir)
        
    documents = load_documents(settings.data_dir)
    chunk = split_documents(documents)
    
    embedding = OpenAIEmbeddings(model=settings.embedding_model)
    vector_store = Chroma(
        collection_name=settings.collection_name,
        embedding_function=embedding,
        persist_directory= str(settings.persist_dir)
        
    )
    
    ids = [str(uuid4()) for _ in chunk]
    
    vector_store.add_documents(documents=chunk, ids=ids)
    
    return {
        "document_count": len(documents),
        "chunk_count": len(chunk),
        "collection_name": settings.collection_name,
        "persist_dir": str(settings.persist_dir)
        
    }
    

def main() -> None:
    parser = argparse.ArgumentParser(description="Chroma vector index build")
    parser.add_argument(
        "--reset",
        action = "store_true", # -> False -> True
        help = "Delete the old chroma database before indexing documents"
    )
    args = parser.parse_args()
    
    result = build_index(reset=args.reset)
    
    print("Indexing Created! ")
    
if __name__ == "__main__":
    main()