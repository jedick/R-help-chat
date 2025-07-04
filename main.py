from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage.file_system import LocalFileStore
from bm25s_retriever import BM25SRetriever
from contextlib import contextmanager
from datetime import datetime
import chromadb
import os
import glob
import sys

# R-help-chat
# First version by Jeffrey Dick on 2025-06-29

# TODO: use UUID to process only changed documents
# https://stackoverflow.com/questions/76265631/chromadb-add-single-document-only-if-it-doesnt-exist

# Path for input files
path = "R-help"

# Collection name and persistent directory for ChromaDB
collection_name = "R-help2"
persist_directory = f"db/chroma/{collection_name}"
# File store for ParentDocumentRetriever
file_store = "db/file_store"
# BM25 persistent directory
bm25_persist_directory = f"db/bm25/{collection_name}"


@contextmanager
def suppress_stderr():
    """
    Context for suppressing stderr messages
    """
    try:
        # Save the original stderr
        original_stderr = sys.stderr
        sys.stderr = open(os.devnull, "w")  # Redirect to null device
        yield  # Code inside the `with` block executes here
    finally:
        # Restore stderr
        sys.stderr = original_stderr


def build_retriever(search_type: str = "hybrid"):
    """
    Build retriever instance
    
    Args:
        search_type: Type of search to use. Options: "dense", "sparse", "hybrid"
    """
    if search_type == "sparse":
        # Use BM25 sparse search
        retriever = BM25SRetriever.from_persisted_directory(
            path=bm25_persist_directory,
            k=10,
        )
        return retriever
    elif search_type == "dense":
        # Use dense vector search with ChromaDB
        # Define embedding model
        embedding = OpenAIEmbeddings(
            # api_key=openai_api_key,
            model="text-embedding-3-small",
        )
        # Create vector store, suppressing messages:
        # Failed to send telemetry event ClientStartEvent: capture() takes 1 positional argument but 3 were given
        # Failed to send telemetry event ClientCreateCollectionEvent: capture() takes 1 positional argument but 3 were given
        with suppress_stderr():
            vectorstore = Chroma(
                collection_name=collection_name,
                persist_directory=persist_directory,
                embedding_function=embedding,
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
        parent_splitter = RecursiveCharacterTextSplitter(separators=["\n\nFrom"])
        # Instantiate a retriever
        retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=byte_store,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
        )
        return retriever
    else:
        # Hybrid search - for now, default to dense search
        # TODO: Implement hybrid search combining dense and sparse results
        return build_retriever("dense")


def ProcessDirectory(path, search_type: str = "hybrid"):
    """
    Update vector store for files in a directory, only adding new or updated files
    "path": directory to process
    "search_type": Type of search to use. Options: "dense", "sparse", "hybrid"
    """

    # Get a retriever instance
    retriever = build_retriever(search_type)
    
    if search_type == "sparse":
        # For sparse search, we need to handle BM25 differently
        # BM25 doesn't have the same metadata tracking as ChromaDB
        # For now, we'll process all files when using sparse search
        file_paths = glob.glob(f"{path}/*.txt")
        for file_path in file_paths:
            ProcessFile(file_path, search_type)
            print(f"Processed {file_path} for sparse search")
        return
    
    # For dense search, use the existing logic
    # List all text files in target directory
    file_paths = glob.glob(f"{path}/*.txt")
    # Loop over files
    for file_path in file_paths:
        # Look for existing embeddings for this file
        if hasattr(retriever, 'vectorstore') and hasattr(retriever.vectorstore, 'get'):
            results = retriever.vectorstore.get(
                # Metadata key-value pair
                where={"source": file_path}
            )
            # Flag to add or update file
            add_file = False
            update_file = False
            # If file path doesn't exist in vector store, then add it
            if len(results["ids"]) == 0:
                add_file = True
            else:
                # Check file timestamp to decide whether to update embeddings
                mod_time = os.path.getmtime(file_path)
                timestamp = datetime.fromtimestamp(mod_time).isoformat()
                # Loop over metadata and compare to actual file timestamp
                for metadata in results["metadatas"]:
                    # Process file if any of embeddings has a different timestamp
                    if not metadata["timestamp"] == timestamp:
                        add_file = True
                        break
                # Delete the old embeddings
                if add_file:
                    with suppress_stderr():
                        retriever.vectorstore.delete(results["ids"])
                    update_file = True

            if add_file:
                ProcessFile(file_path, search_type)

            if update_file:
                print(f"Updated embeddings for {file_path}")
                # Clear out the unused parent files
                # The used doc_ids are the files to keep
                used_doc_ids = [
                    d["doc_id"] for d in retriever.vectorstore.get()["metadatas"]
                ]
                files_to_keep = list(set(used_doc_ids))
                # Get all files in the file store
                all_files = os.listdir(file_store)
                # Iterate through the files and delete those not in the list
                for file in all_files:
                    file_path = os.path.join(file_store, file)
                    if os.path.isfile(file_path) and file not in files_to_keep:
                        os.remove(file_path)
            elif add_file:
                print(f"Added embeddings for {file_path}")
            else:
                print(f"No change for {file_path}")
        else:
            # Fallback for other retriever types
            ProcessFile(file_path, search_type)
            print(f"Processed {file_path}")


def ProcessFile(file_path, search_type: str = "hybrid"):
    """
    Splits file into chunks and saves vector embeddings
    "file_path": file to process
    "search_type": Type of search to use. Options: "dense", "sparse", "hybrid"
    """

    if search_type == "sparse":
        # Handle sparse search with BM25
        ProcessFileSparse(file_path)
    else:
        # Handle dense search with ChromaDB
        ProcessFileDense(file_path)


def ProcessFileDense(file_path):
    """
    Process file for dense vector search using ChromaDB
    "file_path": file to process
    """
    # Get a retriever instance
    retriever = build_retriever("dense")
    # Load text file to document
    loader = TextLoader(file_path)
    document = loader.load()
    # Add documents to vectorstore
    with suppress_stderr():
        retriever.add_documents(document)

    # Add file timestamps to metadata
    with suppress_stderr():
        AddTimestamps(file_path)


def ProcessFileSparse(file_path):
    """
    Process file for sparse search using BM25
    "file_path": file to process
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
        retriever = BM25SRetriever.from_persisted_directory(bm25_persist_directory)
        # Get new emails - ones which have not been indexed
        new_emails = [email for email in emails if email not in retriever.docs]
        # TODO: implement add_documents method for BM25SRetriever class
        # If add_documents was available, we could just index the new emails
        #retriever.from_documents(documents=new_emails)
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
            with suppress_stderr():
                collection.update(id, metadatas=metadata)


def ListDocuments():

    # Get retriever instance
    retriever = build_retriever()
    # Retrieve all document IDs
    with suppress_stderr():
        document_ids = retriever.vectorstore.get()["ids"]
    # Return the document IDs
    return document_ids


def QueryDatabase(query, search_type: str = "hybrid"):
    """
    Retrieve documents from database and query with LLM

    Example: QueryDatabase("What R functions are discussed?")
    """

    # Get retriever instance
    retriever = build_retriever(search_type)
    
    if retriever is None:
        return "No retriever available. Please process some documents first."

    # Creating a prompt template
    prompt = ChatPromptTemplate.from_template(
        """Use only the provided context to answer the following question.
        If the context does not have enough information to answer the question, say so.
        Context: {context}
        Question: {question}
        """
    )
    # Define a model
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Building an LCEL retrieval chain
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Invoking the retrieval chain
    with suppress_stderr():
        result = chain.invoke(query)
    return result
