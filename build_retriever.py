from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain.retrievers import ParentDocumentRetriever, EnsembleRetriever
from langchain.storage.file_system import LocalFileStore
from datetime import datetime
import chromadb
import os

# To use OpenAI models (remote)
from langchain_openai import OpenAIEmbeddings

## To use Hugging Face models (local)
# from langchain_huggingface import HuggingFaceEmbeddings
# For more control over BGE and Nomic embeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

# Local modules
from bm25s_retriever import BM25SRetriever

# Collection name and persistent directory for ChromaDB
collection_name = "January-2025"
persist_directory = f"db/chroma/{collection_name}"
# File store for ParentDocumentRetriever
file_store = f"db/file_store/{collection_name}"
# BM25 persistent directory
bm25_persist_directory = f"db/bm25/{collection_name}"


def GetRetrieverParam(param_name: str):
    """
    Get a parameter define in this file

    Args:
        param_name: Name of parameter, e.g. "file_store"
    """
    # Return a variable named in param_name for use outside the module
    return globals()[param_name]


def BuildRetriever(search_type: str = "hybrid", embedding_api: str = "local"):
    """
    Build retriever instance

    Args:
        search_type: Type of search to use. Options: "dense", "sparse", "hybrid"
        embedding_api: Type of embedding API (remote or local)
    """
    if search_type == "sparse":
        return BuildRetrieverSparse()
    elif search_type == "dense":
        return BuildRetrieverDense(embedding_api=embedding_api)
    elif search_type == "hybrid":
        # Hybrid search - use ensemble method
        # https://python.langchain.com/api_reference/langchain/retrievers/langchain.retrievers.ensemble.EnsembleRetriever.html
        sparse_retriever = BuildRetrieverSparse()
        dense_retriever = BuildRetrieverDense(embedding_api=embedding_api)
        ensemble_retriever = EnsembleRetriever(
            retrievers=[sparse_retriever, dense_retriever], weights=[0.5, 0.5]
        )
        return ensemble_retriever
    else:
        raise ValueError(f"Unsupported search type: f{search_type}")


def BuildRetrieverSparse():
    """
    Build sparse retriever instance
    """
    if not os.path.exists(bm25_persist_directory):
        os.makedirs(bm25_persist_directory)

    # Use BM25 sparse search
    retriever = BM25SRetriever.from_persisted_directory(
        path=bm25_persist_directory,
        k=10,
    )
    return retriever


def BuildRetrieverDense(embedding_api: str = "local"):
    """
    Build dense retriever instance

    Args:
        embedding_api: Type of embedding API (remote or local)
    """
    # Use dense vector search with ChromaDB
    # Define embedding model
    if embedding_api == "remote":
        embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")
    if embedding_api == "local":
        # embedding_function = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5", show_progress=True)
        # https://python.langchain.com/api_reference/community/embeddings/langchain_community.embeddings.huggingface.HuggingFaceBgeEmbeddings.html
        model_name = "nomic-ai/nomic-embed-text-v1.5"
        model_kwargs = {
            "device": "cuda",
            "trust_remote_code": True,
        }
        encode_kwargs = {"normalize_embeddings": True}
        embedding_function = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
            query_instruction="search_query:",
            embed_instruction="search_document:",
        )
    # Create vector store
    vectorstore = Chroma(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_function=embedding_function,
    )
    # The storage layer for the parent documents
    byte_store = LocalFileStore(file_store)
    # Text splitter for child documents
    child_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " ", ""],
        chunk_size=1000,
        chunk_overlap=100,
    )
    # Text splitter for parent documents
    parent_splitter = RecursiveCharacterTextSplitter(separators=["\n\nFrom"], chunk_size=1, chunk_overlap=0)
    # Instantiate a retriever
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        # NOTE: https://github.com/langchain-ai/langchain/issues/9345
        # Define byte_store = LocalFileStore(file_store) and use byte_store instead of docstore in ParentDocumentRetriever
        byte_store=byte_store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )
    ## Release GPU memory
    ## https://github.com/langchain-ai/langchain/discussions/10668
    # torch.cuda.empty_cache()
    return retriever


def AddTimestamps(file_path):
    """
    Adds timestamps to metadata in vector store.
    "file_path": used for both filtering the vector store and obtaining file modification time

    Usage: AddTimestamps("R-help/2025-January.txt")
    """
    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.get_collection(collection_name)
    # Filter by "source" metadata field (added by DirectoryLoader)
    results = collection.get(where={"source": file_path})
    for id, metadata in zip(results["ids"], results["metadatas"]):
        # Add timestamp if it's not present
        if not "timestamp" in metadata:
            mod_time = os.path.getmtime(file_path)
            timestamp = datetime.fromtimestamp(mod_time).isoformat()
            metadata["timestamp"] = timestamp
            # Update the document in the vector store
            collection.update(id, metadatas=metadata)
