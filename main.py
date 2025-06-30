from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
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

# TODO: Use ParentDocumentRetriever
# During retrieval, it first fetches the small chunks but then looks up the parent ids for those chunks and returns those larger documents.
# https://python.langchain.com/docs/how_to/parent_document_retriever/

# Define path for input files
path = "R-help"

# Define embedding model
embedding = OpenAIEmbeddings(
    # api_key=openai_api_key,
    model="text-embedding-3-small",
)

# Define collection name and persistent directory for ChromaDB
collection_name = "R-help"
persist_directory = "/home/chromadb/R-help"


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
    # Create vector store, suppressing messages:
    # Failed to send telemetry event ClientStartEvent: capture() takes 1 positional argument but 3 were given
    # Failed to send telemetry event ClientCreateCollectionEvent: capture() takes 1 positional argument but 3 were given
    with suppress_stderr():
        vectorstore = Chroma(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_function=embedding,
        )
    # Instantiate a retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10},
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
        # Query documents by metadata
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
            # Get file timestamp
            mod_time = os.path.getmtime(file_path)
            timestamp = datetime.fromtimestamp(mod_time).isoformat()
            # Loop over metadata and compare to actual file timestamp
            for metadata in results["metadatas"]:
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
        elif add_file:
            print(f"Added embeddings for {file_path}")
        else:
            print(f"No change for {file_path}")


def ProcessFile(file_path):
    """
    Splits file into chunks and saves vector embeddings
    "file_path": file to process
    """

    # Text splitter
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " ", ""],
        chunk_size=1000,
        chunk_overlap=100,
    )

    # Splitting documents
    loader = TextLoader(file_path)
    documents = loader.load()
    chunks = splitter.split_documents(documents)
    # Make sure we have at least one chunk
    assert len(chunks) > 0, f"Got no chunks when splitting {file_path}!"
    # print([len(chunk.page_content) for chunk in chunks])
    print(f"Got {len(chunks)} chunks from {file_path}")

    # Get a retriever instance
    retriever = build_retriever()
    # Add documents to vectorstore
    with suppress_stderr():
        retriever.vectorstore.add_documents(chunks)

    # Add file timestamps to metadata
    AddTimestamps(file_path)


def AddTimestamps(file_path):
    """
    Adds timestamps to metadata in vector store.
    "file_path": used for both filtering the vector store and obtaining file modification time

    Usage: AddTimestamps("R-help/2025-January.txt")
    """
    with suppress_stderr():
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
    result = chain.invoke(query)
    return result
