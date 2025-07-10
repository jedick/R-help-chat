from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# Local modules
from bm25s_retriever import BM25SRetriever
from build_retriever import BuildRetriever, GetRetrieverParam, AddTimestamps
from util import SuppressStderr


def ProcessFile(file_path, search_type: str = "dense", embedding_api: str = "local"):
    """
    Splits file into chunks and saves vector embeddings

    Args:
        file_path: file to process
        search_type: Type of search to use. Options: "dense", "sparse"
        embedding_api: Type of embedding API (remote or local)
    """

    if search_type == "sparse":
        # Handle sparse search with BM25
        ProcessFileSparse(file_path)
    elif search_type == "dense":
        # Handle dense search with ChromaDB
        ProcessFileDense(file_path, embedding_api)
    else:
        raise ValueError(f"Unsupported search type: f{search_type}")


def ProcessFileDense(file_path, embedding_api):
    """
    Process file for dense vector search using ChromaDB
    """
    with SuppressStderr():
        # Get a retriever instance
        retriever = BuildRetriever("dense", embedding_api)
        # Load text file to document
        loader = TextLoader(file_path)
        document = loader.load()
        # Add documents to vectorstore
        retriever.add_documents(document)
        # Add file timestamps to metadata
        AddTimestamps(file_path)


def ProcessFileSparse(file_path):
    """
    Process file for sparse search using BM25
    """
    # Load text file to document
    loader = TextLoader(file_path)
    documents = loader.load()

    # Split archive file into emails for BM25
    splitter = RecursiveCharacterTextSplitter(separators=["\n\nFrom"])
    emails = splitter.split_documents(documents)

    # Add source metadata to emails
    for email in emails:
        if "source" not in email.metadata:
            email.metadata["source"] = file_path

    # Create or update BM25 index
    try:
        # Update BM25 index if it exists
        bm25_persist_directory = GetRetrieverParam("bm25_persist_directory")
        retriever = BM25SRetriever.from_persisted_directory(bm25_persist_directory)
        # Get new emails - ones which have not been indexed
        new_emails = [email for email in emails if email not in retriever.docs]
        # TODO: implement add_documents method for BM25SRetriever class
        # If add_documents was available, we could just index the new emails
        # retriever.from_documents(documents=new_emails)
        # For now, create new BM25 index with all emails
        all_emails = retriever.docs + new_emails
        BM25SRetriever.from_documents(
            documents=emails,
            persist_directory=bm25_persist_directory,
        )
        print(f"BM25 index updated with {len(new_emails)} emails from {file_path}")
    except (FileNotFoundError, OSError):
        # Create new BM25 index
        BM25SRetriever.from_documents(
            documents=emails,
            persist_directory=bm25_persist_directory,
        )
        print(f"BM25 index created with {len(emails)} emails from {file_path}")
