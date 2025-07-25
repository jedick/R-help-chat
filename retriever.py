# Main retriever modules
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain.retrievers import ParentDocumentRetriever, EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever, RetrieverLike
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from typing import Any, Optional
import chromadb
import torch
import os
import re

# To use OpenAI models (remote)
from langchain_openai import OpenAIEmbeddings

## To use Hugging Face models (local)
# from langchain_huggingface import HuggingFaceEmbeddings
# For more control over BGE and Nomic embeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

# Local modules
from mods.bm25s_retriever import BM25SRetriever
from mods.file_system import LocalFileStore

# Database directory
db_dir = "db"


def BuildRetriever(
    compute_mode,
    search_type: str = "hybrid",
    top_k=6,
    start_year=None,
    end_year=None,
):
    """
    Build retriever instance.
    All retriever types are configured to return up to 6 documents for fair comparison in evals.

    Args:
        compute_mode: Compute mode for embeddings (remote or local)
        search_type: Type of search to use. Options: "dense", "sparse", "hybrid"
        top_k: Number of documents to retrieve for "dense" and "sparse"
        start_year: Start year (optional)
        end_year: End year (optional)
    """
    if search_type == "dense":
        if not (start_year or end_year):
            # No year filtering, so directly use base retriever
            return BuildRetrieverDense(compute_mode, top_k=top_k)
        else:
            # Get 1000 documents then keep top_k filtered by year
            base_retriever = BuildRetrieverDense(compute_mode, top_k=1000)
            return TopKRetriever(
                base_retriever=base_retriever,
                top_k=top_k,
                start_year=start_year,
                end_year=end_year,
            )
    if search_type == "sparse":
        if not (start_year or end_year):
            return BuildRetrieverSparse(top_k=top_k)
        else:
            base_retriever = BuildRetrieverSparse(top_k=1000)
            return TopKRetriever(
                base_retriever=base_retriever,
                top_k=top_k,
                start_year=start_year,
                end_year=end_year,
            )
    elif search_type == "hybrid":
        # Hybrid search (dense + sparse) - use ensemble method
        # https://python.langchain.com/api_reference/langchain/retrievers/langchain.retrievers.ensemble.EnsembleRetriever.html
        # Use floor (top_k // 2) and ceiling -(top_k // -2) to divide odd values of top_k
        # https://stackoverflow.com/questions/14822184/is-there-a-ceiling-equivalent-of-operator-in-python
        dense_retriever = BuildRetriever(
            compute_mode, "dense", (top_k // 2), start_year, end_year
        )
        sparse_retriever = BuildRetriever(
            compute_mode, "sparse", -(top_k // -2), start_year, end_year
        )
        ensemble_retriever = EnsembleRetriever(
            retrievers=[dense_retriever, sparse_retriever], weights=[1, 1]
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


def BuildRetrieverDense(compute_mode: str, top_k=6):
    """
    Build dense retriever instance with ChromaDB vectorstore

    Args:
        compute_mode: Compute mode for embeddings (remote or local)
        top_k: Number of documents to retrieve
    """

    # Don't try to use local models without a GPU
    if compute_mode == "local" and not torch.cuda.is_available():
        raise Exception("Local embeddings selected without GPU")

    # Define embedding model
    if compute_mode == "remote":
        embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")
    if compute_mode == "local":
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
    persist_directory = f"{db_dir}/chroma_{compute_mode}"
    vectorstore = Chroma(
        collection_name="R-help",
        embedding_function=embedding_function,
        client_settings=client_settings,
        persist_directory=persist_directory,
    )
    # The storage layer for the parent documents
    file_store = f"{db_dir}/file_store_{compute_mode}"
    byte_store = LocalFileStore(file_store)
    # Text splitter for child documents
    child_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " ", ""],
        chunk_size=1000,
        chunk_overlap=100,
    )
    # Text splitter for parent documents
    parent_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n\nFrom"], chunk_size=1, chunk_overlap=0
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


class TopKRetriever(BaseRetriever):
    """Retriever that wraps a base retriever and returns the top k documents, optionally matching given start and/or end years."""

    # Code adapted from langchain/retrievers/contextual_compression.py

    base_retriever: RetrieverLike
    """Base Retriever to use for getting relevant documents."""

    top_k: int = 6
    """Number of documents to return."""

    start_year: Optional[int] = None
    end_year: Optional[int] = None

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> list[Document]:
        """Return the top k documents within start and end years if given.

        Returns:
            Sequence of documents
        """
        # Run the search with the base retriever
        filtered_docs = retrieved_docs = self.base_retriever.invoke(
            query, config={"callbacks": run_manager.get_child()}, **kwargs
        )
        if retrieved_docs:

            # Get the sources (file names) and years
            sources = [doc.metadata["source"] for doc in filtered_docs]
            years = [
                re.sub(r"-[A-Za-z]+\.txt", "", source.replace("R-help/", ""))
                for source in sources
            ]
            # Convert years to integer
            years = [int(year) for year in years]

            # Filtering by year
            if self.start_year:
                in_range = after_start = [year >= self.start_year for year in years]
            if self.end_year:
                in_range = before_end = [year <= self.end_year for year in years]
            if self.start_year and self.end_year:
                in_range = [
                    after and before for after, before in zip(after_start, before_end)
                ]
            if self.start_year or self.end_year:
                # Extract docs where the year is in the start-end range
                filtered_docs = [
                    doc for doc, in_range in zip(retrieved_docs, in_range) if in_range
                ]

            # Return the top k docs
            return filtered_docs[: self.top_k]

        else:
            return []
