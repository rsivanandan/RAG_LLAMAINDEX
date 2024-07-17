# All the RAG - Document Ingestion & Retrieval code

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from IPython.display import Markdown, display
from llama_index.llms.ollama import Ollama
from llama_index.core.chat_engine.types import ChatMode
from llama_index.llms.huggingface import HuggingFaceInferenceAPI
import chromadb
import streamlit as st

@st.cache_resource(show_spinner=False)
def init_llm_ollama():
    llm = Ollama(model="llama3", request_timeout=600.0)
    
    # define embedding function

    embed_model = OllamaEmbedding(
        model_name="llama3"
    )
    Settings.llm = llm
    Settings.embed_model = embed_model

@st.cache_resource(show_spinner=False)
def init_index() -> VectorStoreIndex:
    # Load from disk

    load_client = chromadb.PersistentClient(path="./chroma_db")

    # Fetch the collection
    chroma_collection = load_client.get_collection("tsr_collection")

    # Fetch the vector store

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # Get the index from the vector store

    return VectorStoreIndex.from_vector_store(vector_store)


@st.cache_resource(show_spinner=False)
def Create_Vector() -> None:

    # create client and a new collection

    chroma_client = chromadb.EphemeralClient()

    chroma_client.get_or_create_collection(name="tsr_collection")
    chroma_client.delete_collection("tsr_collection")

    embed_model = OllamaEmbedding(
        model_name="llama3"
    )

    # load documents
    documents = SimpleDirectoryReader("./data").load_data()

    # save vector & save to disk

    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db.get_or_create_collection("tsr_collection")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, embed_model=embed_model
    )


def Create_Graph() -> None:
    st.info("Not Implemented Yet")
