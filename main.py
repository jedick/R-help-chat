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


def build_retriever():
    """Build retriever instance"""
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
        byte_store=byte_store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    return retriever


def ProcessDirectory(path):
    """
    Update vector store for files in a directory, only adding new or updated files
    "path": directory to process
    """

    # Get a retriever instance
    retriever = build_retriever()
    # List all text files in target directory
    file_paths = glob.glob(f"{path}/*.txt")
    # Loop over files
    for file_path in file_paths:
        # Look for existing embeddings for this file
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
            ProcessFile(file_path)

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


def ProcessFile(file_path):
    """
    Splits file into chunks and saves vector embeddings
    "file_path": file to process
    """

    # Splitting documents
    # chunks = splitter.split_documents(documents)
    ## Make sure we have at least one chunk
    # assert len(chunks) > 0, f"Got no chunks when splitting {file_path}!"
    ## print([len(chunk.page_content) for chunk in chunks])
    # print(f"Got {len(chunks)} chunks from {file_path}")

    # Get a retriever instance
    retriever = build_retriever()
    # Load text file to document
    loader = TextLoader(file_path)
    document = loader.load()
    # Add documents to vectorstore
    with suppress_stderr():
        retriever.add_documents(document)

    # Add file timestamps to metadata
    with suppress_stderr():
        AddTimestamps(file_path)


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


def QueryDatabase(query):
    """
    Retrieve documents from database and query with LLM

    Example: QueryDatabase("What R functions are discussed?")
    """

    # Get retriever instance
    retriever = build_retriever()

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
