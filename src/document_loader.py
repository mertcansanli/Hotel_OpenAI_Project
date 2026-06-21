from pathlib import Path 

from typing import Iterable, List

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

main_extensions = {".txt",".pdf",".md"}

def iter_document_path(data_dir:Path) -> Iterable[Path]:
    for path in sorted(data_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in main_extensions:
            yield path
        
def load_single_file(file_path: Path) -> Document:
    if not file_path.is_file():
        raise ValueError("Provided file path is not a file")
    
    if file_path.suffix.lower() not in main_extensions:
        raise ValueError(" This extensions is not supported, please try a diffrent format")
    
    if file_path.suffix.lower() == ".pdf":
        loader= PyPDFLoader(str(file_path), encoding = "utf-8")
        
    elif file_path.suffix.lower() in {".txt",".md"}:
        loader = TextLoader(str(file_path), encoding="utf-8")
        
    else : 
        return []
    
    documents =loader.load()
    
    for doc in documents: 
        doc.metadata["source"] = str(file_path)
        doc.metadata["file_name"] = file_path.name
        
    return documents

def load_documents(data_dir: Path) -> List[Document]:
    if not data_dir.exists():
        raise ValueError("Provided data directory does not exist!")
    
    all_documents: List[Document] = []
    
    for path in iter_document_path(data_dir):
        documents = load_single_file(path)
        all_documents.extend(documents)
    return all_documents
        