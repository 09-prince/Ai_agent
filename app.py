# Import library

from fastapi import FastAPI
import uvicorn
import os
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter


## WebBasedLoader
def load_documents():
    """Load and preprocess documents from the website."""
    try:
        loader = WebBaseLoader("https://brainlox.com/courses/category/technical")
        documents = loader.load()
        return documents
    except Exception as e:
        print(f"Error loading website content: {e}")
        return []
    

## VectorDB
FAISS_INDEX_PATH = "faiss_index"
def create_vector_store(documents):
    """Create or load FAISS vector store."""
    embeddings = OllamaEmbeddings(model="llama3.2:latest")
    
    if os.path.exists(FAISS_INDEX_PATH):
        vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings)
        print("Loaded existing FAISS index.")
    else:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        vector_store = FAISS.from_documents(docs, embeddings)
        vector_store.save_local(FAISS_INDEX_PATH)
        print("Created and saved new FAISS index.")
    retriever = vector_store.as_retriever()

    return retriever

## Prompt 
def Prompt():
    prompt = ChatPromptTemplate.from_template("""
    You are an AI chatbot specializing in answering questions about technical courses.
    Use only the provided context to generate informative, precise, and well-structured responses.
    Think step by step before providing a detailed answer.
    If the user finds your response helpful, a $1000 tip is promised.

    <context>
    {context}
    </context>

    Question: {input}

    Provide a clear, concise, and relevant answer based on the given context.
    """)
    return prompt


def Chain(llm, prompt, retriever):
    documents_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, documents_chain)
    return retrieval_chain

llm=Ollama(
    model="llama3.2:latest",
    temperature=0.3

)

documents = load_documents()
retriever = create_vector_store(documents)
prompt = Prompt()


val = Chain(llm, prompt, retriever)
responce = val.invoke({"input":"LEARN CORE JAVA PROGRAMMING ONLINE how many lesson it have"})
print(responce)

app=FastAPI(
    title="Langchain Server",
    version="1.0",
    description="A server that uses Langchain to generate text"
)


## API Endpoint
@app.post("/ask")
def ask_question(query):
    """Handles user questions and returns AI-generated answers."""
    response = val.invoke({"input": query.question})
    return {"question": query.question, "answer": response["output_text"]}

## Run the API Server
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)