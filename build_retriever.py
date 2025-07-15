from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain.retrievers import ParentDocumentRetriever, EnsembleRetriever
from langchain.storage.file_system import LocalFileStore
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank
from flashrank import Ranker
import chromadb
import torch
import os

# To use OpenAI models (cloud)
from langchain_openai import OpenAIEmbeddings

## To use Hugging Face models (edge)
# from langchain_huggingface import HuggingFaceEmbeddings
# For more control over BGE and Nomic embeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

# Local modules
from bm25s_retriever import BM25SRetriever

# Database directory
db_dir = "db"


def GetRetrieverParam(param_name: str):
    """
    Get a parameter define in this file

    Args:
        param_name: Name of parameter, e.g. "file_store"
    """
    # Return a variable named in param_name for use outside the module
    return globals()[param_name]


def BuildRetriever(compute_location, search_type: str = "hybrid_rr", top_k=6):
    """
    Build retriever instance.
    All retriever types are configured to return up to 6 documents for fair comparison in evals.

    Args:
        compute_location: Compute location for embeddings (cloud or edge)
        search_type: Type of search to use. Options: "dense", "sparse", "sparse_rr", "hybrid", "hybrid_rr"
        top_k: Number of documents to retrieve for "dense", "sparse", and "sparse_rr"
    """
    if search_type == "dense":
        return BuildRetrieverDense(compute_location, top_k)
    if search_type == "sparse":
        # This gets top_k documents
        sparse_retriever = BuildRetrieverSparse(top_k)
        return sparse_retriever
    if search_type == "sparse_rr":
        # Start with 10 documents
        sparse_retriever = BuildRetrieverSparse(10)
        # Reranking
        client = Ranker(model_name="ms-marco-MultiBERT-L-12", max_length=10000)
        # Keep top_k documents
        compressor = FlashrankRerank(client=client, top_n=top_k)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=sparse_retriever
        )
        return compression_retriever
    elif search_type == "hybrid":
        # Hybrid search (dense + sparse) - use ensemble method
        # https://python.langchain.com/api_reference/langchain/retrievers/langchain.retrievers.ensemble.EnsembleRetriever.html
        # Combine 2 retrievers with 3 docs each (6 docs - fair comparison with hybrid_rr search)
        dense_retriever = BuildRetriever(compute_location, "dense", top_k=3)
        sparse_retriever = BuildRetriever(compute_location, "sparse", top_k=3)
        ensemble_retriever = EnsembleRetriever(
            retrievers=[dense_retriever, sparse_retriever], weights=[1, 1]
        )
        return ensemble_retriever
    elif search_type == "hybrid_rr":
        # Hybrid search (dense + sparse + sparse_rr)
        # Combine 3 retrievers with 2 docs each (6 docs - fair comparison with hybrid search)
        sparse_retriever = BuildRetriever(compute_location, "sparse", top_k=2)
        sparse_rr_retriever = BuildRetriever(compute_location, "sparse_rr", top_k=2)
        dense_retriever = BuildRetriever(compute_location, "dense", top_k=2)
        ensemble_retriever = EnsembleRetriever(
            retrievers=[dense_retriever, sparse_retriever, sparse_rr_retriever],
            weights=[1, 1, 1],
        )
        return ensemble_retriever
    else:
        raise ValueError(f"Unsupported search type: {search_type}")


def BuildRetrieverSparse(top_k=6):
    """
    Build sparse retriever instance

    Args:
        top_k: Number of documents to retrieve
    """
    # BM25 persistent directory
    bm25_persist_directory = f"{db_dir}/bm25"
    if not os.path.exists(bm25_persist_directory):
        os.makedirs(bm25_persist_directory)

    # Use BM25 sparse search
    retriever = BM25SRetriever.from_persisted_directory(
        path=bm25_persist_directory,
        k=top_k,
    )
    return retriever


def BuildRetrieverDense(compute_location: str, top_k=6):
    """
    Build dense retriever instance with ChromaDB vectorstore

    Args:
        compute_location: Compute location for embeddings (cloud or edge)
        top_k: Number of documents to retrieve
    """

    # Don't try to use edge models without a GPU
    if compute_location == "edge" and not torch.cuda.is_available():
        raise Exception("Edge embeddings selected without GPU")

    # Define embedding model
    if compute_location == "cloud":
        embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")
    if compute_location == "edge":
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
    client_settings = chromadb.config.Settings(anonymized_telemetry=False)
    persist_directory = f"{db_dir}/chroma_{compute_location}"
    vectorstore = Chroma(
        collection_name="R-help",
        embedding_function=embedding_function,
        client_settings=client_settings,
        persist_directory=persist_directory,
    )
    # The storage layer for the parent documents
    file_store = f"{db_dir}/file_store_{compute_location}"
    byte_store = LocalFileStore(file_store)
    # Text splitter for child documents
    child_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " ", ""],
        chunk_size=1000,
        chunk_overlap=100,
    )
    # Text splitter for parent documents
    parent_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\nFrom"], chunk_size=1, chunk_overlap=0
    )
    # Instantiate a retriever
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        # NOTE: https://github.com/langchain-ai/langchain/issues/9345
        # Define byte_store = LocalFileStore(file_store) and use byte_store instead of docstore in ParentDocumentRetriever
        byte_store=byte_store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        # Get top k documents
        search_kwargs={"k": top_k},
    )
    ## Release GPU memory
    ## https://github.com/langchain-ai/langchain/discussions/10668
    # torch.cuda.empty_cache()
    return retriever
