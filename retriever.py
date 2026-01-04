# Main retriever modules
from langchain_classic.retrievers import ParentDocumentRetriever, EnsembleRetriever
from langchain_core.retrievers import BaseRetriever, RetrieverLike
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from typing import Any, Optional
import chromadb
import os
import re
from calendar import month_abbr, month_name

# Local modules
from mods.bm25s_retriever import BM25SRetriever
from mods.file_system import LocalFileStore
from util import get_sources


def BuildRetriever(
    db_dir: str,
    collection: str,
    search_type: str,
    top_k: Optional[int] = 6,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    months: Optional[list[str]] = None,
):
    """
    Build retriever instance.
    All retriever types are configured to return up to 6 documents for fair comparison in evals.

    Args:
        db_dir: Database directory
        collection: Email collection
        search_type: Type of search to use. Options: "dense", "sparse", "hybrid"
        top_k: Number of documents to retrieve for "dense" and "sparse"
        start_year: Start year (optional)
        end_year: End year (optional)
        months: List of months (3-letter abbreviations) (optional)
    """
    if search_type == "dense":
        if not (start_year or end_year or months):
            # No year or month filtering, so directly use base retriever
            return BuildRetrieverDense(
                db_dir=db_dir, collection=collection, top_k=top_k
            )
        else:
            # Get 10000 documents then keep top_k filtered by year and month
            base_retriever = BuildRetrieverDense(
                db_dir=db_dir, collection=collection, top_k=10000
            )
            return TopKRetriever(
                base_retriever=base_retriever,
                top_k=top_k,
                start_year=start_year,
                end_year=end_year,
                months=months,
            )
    if search_type == "sparse":
        if not (start_year or end_year or months):
            return BuildRetrieverSparse(
                db_dir=db_dir, collection=collection, top_k=top_k
            )
        else:
            base_retriever = BuildRetrieverSparse(
                db_dir=db_dir, collection=collection, top_k=10000
            )
            return TopKRetriever(
                base_retriever=base_retriever,
                top_k=top_k,
                start_year=start_year,
                end_year=end_year,
                months=months,
            )
    elif search_type == "hybrid":
        # Hybrid search (dense + sparse) - use ensemble method
        # https://python.langchain.com/api_reference/langchain/retrievers/langchain.retrievers.ensemble.EnsembleRetriever.html
        # Use floor (top_k // 2) and ceiling -(top_k // -2) to divide odd values of top_k
        # https://stackoverflow.com/questions/14822184/is-there-a-ceiling-equivalent-of-operator-in-python
        dense_retriever = BuildRetriever(
            db_dir,
            collection,
            "dense",
            (top_k // 2),
            start_year,
            end_year,
            months,
        )
        sparse_retriever = BuildRetriever(
            db_dir,
            collection,
            "sparse",
            -(top_k // -2),
            start_year,
            end_year,
            months,
        )
        ensemble_retriever = EnsembleRetriever(
            retrievers=[dense_retriever, sparse_retriever], weights=[1, 1]
        )
        return ensemble_retriever
    else:
        raise ValueError(f"Unsupported search type: {search_type}")


def BuildRetrieverSparse(db_dir, collection, top_k=6):
    """
    Build sparse retriever instance

    Args:
        db_dir: Database directory
        collection: Email collection
        top_k: Number of documents to retrieve
    """
    # BM25 persistent directory
    bm25_persist_directory = os.path.join(db_dir, collection, "bm25")
    if not os.path.exists(bm25_persist_directory):
        os.makedirs(bm25_persist_directory)

    # Use BM25 sparse search
    # top_k can't be larger than the corpus size (number of emails)
    corpus_size = len(get_sources(db_dir, collection))
    k = top_k if top_k < corpus_size else corpus_size
    retriever = BM25SRetriever.from_persisted_directory(
        path=bm25_persist_directory,
        k=k,
    )
    return retriever


def BuildRetrieverDense(db_dir, collection, top_k=6):
    """
    Build dense retriever instance with ChromaDB vectorstore

    Args:
        db_dir: Database directory
        collection: Email collection
        top_k: Number of documents to retrieve
    """

    # Define embedding model
    embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")
    # Create vector store
    client_settings = chromadb.config.Settings(anonymized_telemetry=False)
    persist_directory = os.path.join(db_dir, collection, "chroma")
    vectorstore = Chroma(
        collection_name=collection,
        embedding_function=embedding_function,
        client_settings=client_settings,
        persist_directory=persist_directory,
    )
    # The storage layer for the parent documents
    file_store = os.path.join(db_dir, collection, "file_store")
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
    return retriever


class TopKRetriever(BaseRetriever):
    """
    Retriever that wraps a base retriever and returns the top k documents,
    optionally matching given start and/or end years.

    Code adapted from langchain/retrievers/contextual_compression.py
    """

    # Base Retriever to use for getting relevant documents
    base_retriever: RetrieverLike
    # Number of documents to return
    top_k: int = 6
    # Optional year and month arguments
    start_year: Optional[int] = None
    end_year: Optional[int] = None
    months: Optional[list[str]] = None

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> list[Document]:
        """
        Return the top k documents within start and end years (and months) if given.

        Returns:
            Sequence of documents
        """
        # Run the search with the base retriever
        filtered_docs = retrieved_docs = self.base_retriever.invoke(
            query, config={"callbacks": run_manager.get_child()}, **kwargs
        )
        if retrieved_docs:

            # Get the email source files and basenames
            sources = [doc.metadata["source"] for doc in filtered_docs]
            filenames = [os.path.basename(source) for source in sources]
            # Get the years and months
            pattern = re.compile(r"(\d{4})-([A-Za-z]+)\.txt")
            matches = [pattern.match(filename) for filename in filenames]
            # Extract years and month names, handling None matches
            years = []
            month_names = []
            for match in matches:
                if match:
                    years.append(int(match.group(1)))
                    month_names.append(match.group(2))
                else:
                    years.append(None)
                    month_names.append(None)

            # Create mapping from 3-letter abbreviations to full month names
            # month_abbr[0] is empty string, month_abbr[1] is "Jan", etc.
            # month_name[0] is empty string, month_name[1] is "January", etc.
            abbr_to_full = {month_abbr[i].lower(): month_name[i] for i in range(1, 13)}

            # Convert months list (3-letter abbreviations) to full month names
            target_months = None
            if self.months:
                target_months = [
                    abbr_to_full.get(month.lower()) for month in self.months
                ]
                # Filter out None values in case of invalid abbreviations
                target_months = [m for m in target_months if m is not None]

            # Initialize filter flags
            year_filter = None
            month_filter = None

            # Filtering by year
            if self.start_year or self.end_year:
                if self.start_year and self.end_year:
                    year_filter = [
                        year is not None
                        and year >= self.start_year
                        and year <= self.end_year
                        for year in years
                    ]
                elif self.start_year:
                    year_filter = [
                        year is not None and year >= self.start_year for year in years
                    ]
                elif self.end_year:
                    year_filter = [
                        year is not None and year <= self.end_year for year in years
                    ]

            # Filtering by month
            if target_months:
                month_filter = [
                    month_name is not None and month_name in target_months
                    for month_name in month_names
                ]

            # Combine filters
            if year_filter is not None and month_filter is not None:
                # Both year and month filters
                combined_filter = [
                    year and month for year, month in zip(year_filter, month_filter)
                ]
                filtered_docs = [
                    doc for doc, keep in zip(retrieved_docs, combined_filter) if keep
                ]
            elif year_filter is not None:
                # Only year filter
                filtered_docs = [
                    doc for doc, keep in zip(retrieved_docs, year_filter) if keep
                ]
            elif month_filter is not None:
                # Only month filter
                filtered_docs = [
                    doc for doc, keep in zip(retrieved_docs, month_filter) if keep
                ]

            # Return the top k docs
            return filtered_docs[: self.top_k]

        else:
            return []
