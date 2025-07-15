# For BM25SRetriever
from __future__ import annotations
import json
from typing import Any, Callable, Dict, Iterable, List, Optional, Union
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

# Main retriever modules
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


class BM25SRetriever(BaseRetriever):
    """`BM25` retriever with `bm25s` backend

    Source: https://github.com/langchain-ai/langchain/pull/28123 by @mspronesti
    """

    vectorizer: Any
    """ BM25S vectorizer."""
    docs: List[Document] = Field(repr=False)
    """List of documents to retrieve from."""
    k: int = 4
    """Number of top results to return"""
    activate_numba: bool = False
    """Accelerate backend"""

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_texts(
        cls,
        texts: Iterable[str],
        metadatas: Optional[Iterable[dict]] = None,
        bm25_params: Optional[Dict[str, Any]] = None,
        stopwords: Union[str, List[str]] = "en",
        stemmer: Optional[Callable[[List[str]], List[str]]] = None,
        persist_directory: str = None,
        **kwargs: Any,
    ) -> BM25SRetriever:
        """
        Create a BM25Retriever from a list of texts.
        Args:
            texts:
                A list of texts to vectorize.
            metadatas:
                A list of metadata dicts to associate with each text.
            bm25_params:
                Parameters to pass to the BM25s vectorizer.
            stopwords:
                The list of stopwords to remove from the text. Defaults to "en".
            stemmer:
                The stemmer to use for stemming the tokens. It is recommended to
                use the PyStemmer library for stemming, but you can also any
                callable that takes a list of strings and returns a list of strings.
            persist_directory:
                The directory to save the BM25 index to.
            **kwargs: Any other arguments to pass to the retriever.

        Returns:
            A BM25SRetriever instance.
        """
        try:
            from bm25s import BM25
            from bm25s import tokenize as bm25s_tokenize
        except ImportError:
            raise ImportError(
                "Could not import bm25s, please install with `pip install " "bm25s`."
            )

        bm25_params = bm25_params or {}
        texts_processed = bm25s_tokenize(
            texts=texts,
            stopwords=stopwords,
            stemmer=stemmer,
            return_ids=False,
            show_progress=False,
        )
        vectorizer = BM25(**bm25_params)
        vectorizer.index(texts_processed)

        metadatas = metadatas or ({} for _ in texts)
        docs = [Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas)]

        persist_directory = persist_directory
        # persist the vectorizer
        vectorizer.save(persist_directory)
        # additionally persist the corpus and the metadata
        with open(f"{persist_directory}/corpus.jsonl", "w") as f:
            for i, d in enumerate(docs):
                entry = {"id": i, "text": d.page_content, "metadata": d.metadata}
                doc_str = json.dumps(entry)
                f.write(doc_str + "\n")

        return cls(vectorizer=vectorizer, docs=docs, **kwargs)

    @classmethod
    def from_documents(
        cls,
        documents: Iterable[Document],
        *,
        bm25_params: Optional[Dict[str, Any]] = None,
        stopwords: Union[str, List[str]] = "en",
        stemmer: Optional[Callable[[List[str]], List[str]]] = None,
        persist_directory: str = None,
        **kwargs: Any,
    ) -> BM25SRetriever:
        """
        Create a BM25Retriever from a list of Documents.
        Args:
            documents:
                A list of Documents to vectorize.
            bm25_params:
                Parameters to pass to the BM25 vectorizer.
            stopwords:
                The list of stopwords to remove from the text. Defaults to "en".
            stemmer:
                The stemmer to use for stemming the tokens. It is recommended to
                use the PyStemmer library for stemming, but you can also any
                callable that takes a list of strings and returns a list of strings.
            persist_directory:
                The directory to save the BM25 index to.
            **kwargs: Any other arguments to pass to the retriever.

        Returns:
            A BM25Retriever instance.
        """
        texts, metadatas = zip(*((d.page_content, d.metadata) for d in documents))
        return cls.from_texts(
            texts=texts,
            metadatas=metadatas,
            bm25_params=bm25_params,
            stopwords=stopwords,
            stemmer=stemmer,
            persist_directory=persist_directory,
            **kwargs,
        )

    @classmethod
    def from_persisted_directory(cls, path: str, **kwargs: Any) -> BM25SRetriever:
        from bm25s import BM25

        vectorizer = BM25.load(path)
        with open(f"{path}/corpus.jsonl", "r") as f:
            corpus = [json.loads(line) for line in f]

        docs = [
            Document(page_content=d["text"], metadata=d["metadata"]) for d in corpus
        ]
        return cls(vectorizer=vectorizer, docs=docs, **kwargs)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        from bm25s import tokenize as bm25s_tokenize

        processed_query = bm25s_tokenize(query, return_ids=False)
        if self.activate_numba:
            self.vectorizer.activate_numba_scorer()
            return_docs = self.vectorizer.retrieve(
                processed_query, k=self.k, backend_selection="numba"
            )
            return [self.docs[i] for i in return_docs.documents[0]]
        else:
            return_docs, scores = self.vectorizer.retrieve(
                processed_query, self.docs, k=self.k
            )
            return [return_docs[0, i] for i in range(return_docs.shape[1])]
