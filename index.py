from langchain_text_splitters import TextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from datetime import datetime
import tempfile
import os

# Local modules
from retriever import BuildRetriever, db_dir
from mods.bm25s_retriever import BM25SRetriever


def ProcessFile(file_path, search_type: str = "dense", compute_mode: str = "cloud"):
    """
    Wrapper function to process file for dense or sparse search

    Args:
        file_path: File to process
        search_type: Type of search to use. Options: "dense", "sparse"
        compute_mode: Compute mode for embeddings (cloud or edge)
    """

    # Preprocess: remove quoted lines and handle email boundaries
    temp_fd, cleaned_temp_file = tempfile.mkstemp(suffix=".txt", prefix="preproc_")
    with open(file_path, "r", encoding="utf-8", errors="ignore") as infile, open(
        cleaned_temp_file, "w", encoding="utf-8"
    ) as outfile:
        for line in infile:
            # Remove lines that start with '>' or whitespace before '>'
            if line.lstrip().startswith(">"):
                continue
            outfile.write(line)
    try:
        os.close(temp_fd)
    except Exception:
        pass

    # Truncate each email at 500 lines and each line to 500 characters
    temp_fd2, truncated_temp_file = tempfile.mkstemp(suffix=".txt", prefix="truncated_")
    with open(cleaned_temp_file, "r", encoding="utf-8") as infile:
        content = infile.read()
    # Split into emails using '\n\n\nFrom' as the separator
    emails = content.split("\n\n\nFrom")
    processed_emails = []
    for i, email in enumerate(emails):
        lines = email.splitlines()
        # Truncate to 500 lines and each line to 500 characters
        truncated_lines = [line[:500] for line in lines[:500]]
        # Add [Email truncated] line to truncated emails
        if len(lines) > len(truncated_lines):
            truncated_lines.append("[Email truncated]")
        processed_emails.append("\n".join(truncated_lines))
    # Join emails back together with '\n\n\nFrom'
    result = "\n\n\nFrom".join(processed_emails)
    # Add two blank lines to the first email so all emails have the same formatting
    # (needed for removing prepended source file names in evals)
    result = "\n\n" + result
    with open(truncated_temp_file, "w", encoding="utf-8") as outfile:
        outfile.write(result)
    try:
        os.close(temp_fd2)
    except Exception:
        pass

    try:
        if search_type == "sparse":
            # Handle sparse search with BM25
            ProcessFileSparse(truncated_temp_file, file_path)
        elif search_type == "dense":
            # Handle dense search with ChromaDB
            ProcessFileDense(truncated_temp_file, file_path, compute_mode)
        else:
            raise ValueError(f"Unsupported search type: {search_type}")
    finally:
        # Clean up the temporary files
        try:
            os.remove(cleaned_temp_file)
            os.remove(truncated_temp_file)
        except Exception:
            pass


def ProcessFileDense(cleaned_temp_file, file_path, compute_mode):
    """
    Process file for dense vector search using ChromaDB
    """
    # Get a retriever instance
    retriever = BuildRetriever(compute_mode, "dense")
    # Load cleaned text file
    loader = TextLoader(cleaned_temp_file)
    documents = loader.load()
    # Use original file path for "source" key in metadata
    documents[0].metadata["source"] = file_path
    # Add file timestamp to metadata
    mod_time = os.path.getmtime(file_path)
    timestamp = datetime.fromtimestamp(mod_time).isoformat()
    documents[0].metadata["timestamp"] = timestamp
    # Add documents to vectorstore
    retriever.add_documents(documents)


def ProcessFileSparse(cleaned_temp_file, file_path):
    """
    Process file for sparse search using BM25
    """
    # Load text file to document
    loader = TextLoader(cleaned_temp_file)
    documents = loader.load()

    # Split archive file into emails for BM25
    # Using two blank lines followed by "From", and no limits on chunk size
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n\nFrom"], chunk_size=1, chunk_overlap=0
    )
    ## Using 'EmailFrom' as the separator (requires preprocesing)
    # splitter = RecursiveCharacterTextSplitter(separators=["EmailFrom"])
    emails = splitter.split_documents(documents)

    # Use original file path for "source" key in metadata
    for email in emails:
        email.metadata["source"] = file_path

    # Create or update BM25 index
    try:
        # Update BM25 index if it exists
        bm25_persist_directory = f"{db_dir}/bm25"
        retriever = BM25SRetriever.from_persisted_directory(bm25_persist_directory)
        # Get new emails - ones which have not been indexed
        new_emails = [email for email in emails if email not in retriever.docs]
        if len(new_emails) > 0:
            # TODO: implement add_documents method for BM25SRetriever class
            # retriever.from_documents(documents=new_emails)
            # For now, create new BM25 index with all emails
            all_emails = retriever.docs + new_emails
            BM25SRetriever.from_documents(
                documents=all_emails,
                persist_directory=bm25_persist_directory,
            )
            print(f"BM25S: added {len(new_emails)} new emails from {file_path}")
        else:
            print(f"BM25S: no change for {file_path}")
    except (FileNotFoundError, OSError):
        # Create new BM25 index
        BM25SRetriever.from_documents(
            documents=emails,
            persist_directory=bm25_persist_directory,
        )
        print(f"BM25S: started with {len(emails)} emails from {file_path}")
