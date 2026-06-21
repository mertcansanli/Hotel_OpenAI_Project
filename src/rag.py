from dataclasses import dataclass

from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from  langchain_openai import ChatOpenAI, OpenAIEmbeddings


from src.config import Settings, required_open_ai_keys

SYSTEM_PROMPT = """
You are a hotel policy assistant for a hotel booking website

Your job is :
- Answer the users question using ONLY the provided hotel policy context.
- Help users understand hotel rules about pets, children, breakfast, parking, check-in, check-out and cancellation.
- Keep the answers practical and concise.
- If multiple hotels match the question, compare them.

Context:
{context}

"""

@dataclass
class RagResponse:
    answer: str
    sources: list[str]
    retrieved_chunks : list[dict]
    
    
def get_vector_store() -> Chroma: 
    required_open_ai_keys()
    settings= Settings()
    embeddings= OpenAIEmbeddings(model=settings.embedding_model)
    
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings,
        persist_directory=str(settings.persist_dir)
    )
    
    
def format_context(docs) -> str:
    
    formatted_chunks = []
    for i, doc in enumerate(docs,start=1):
        file_name= doc.metadata.get("file_name","unkown_source")
        formatted_chunks.append(
            f"[Chunk {i} | Source: {file_name} |]\n{doc.page_content}"
            
        )
    return "\n\n".join(formatted_chunks)

def uniquer_sources(docs) -> list[str]:
    seen= set()
    sources = []
    
    for doc in docs:
        source = doc.metadata.get("file_name","unkown_source")
        if source not in seen:
            sources.append(source)
            seen.add(source)
            
    return sources

def answer_question(question: str, k: int=4) -> RagResponse:
    
    vector_store = get_vector_store()
    retriever= vector_store.as_retriever(search_kwargs={"k":k})
    docs = retriever.invoke(question)
    
    if not docs:
        return RagResponse(
            answer= "There is no data according to your question, please ask an another question.",
            sources = [],
            retrieved_chunks=[]
        )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "Question: {question}"),
            
            
        ]
        
        
    )
    
    settings = Settings()
    llm= ChatOpenAI(model=settings.openai_model,temperature=0)
    chain= prompt | llm | StrOutputParser()
    
    answer = chain.invoke(
        {"context": format_context(docs),
         "question": question,}
    )
    
    retrieved_chunks = [
        {
            
            "source": doc.metadata.get("file_name","unkown_source"),
            "content": doc.page_content,
        }
        for doc in docs
    ]
    
    return RagResponse(
        answer=answer,
        sources= uniquer_sources(docs),
        retrieved_chunks = retrieved_chunks,
    )