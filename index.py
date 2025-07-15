from langchain_text_splitters import TextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from datetime import datetime
import tempfile
import os

# Local modules
from retriever import BuildRetriever, GetRetrieverParam, BM25SRetriever


def ProcessFile(file_path, search_type: str = "dense", compute_location: str = "cloud"):
    """
    Wrapper function to process file for dense or sparse search

    Args:
        file_path: File to process
        search_type: Type of search to use. Options: "dense", "sparse"
        compute_location: Compute location for embeddings (cloud or edge)
    """

    # Preprocess: remove quoted lines and handle email boundaries
    temp_fd, cleaned_temp_file = tempfile.mkstemp(suffix=".txt", prefix="preproc_")
    with open(file_path, "r", encoding="utf-8", errors="ignore") as infile, open(
        cleaned_temp_file, "w", encoding="utf-8"
    ) as outfile:
        prev_line_from = False
        for line in infile:
            # Remove lines that start with '>' or whitespace before '>'
            if line.lstrip().startswith(">"):
                continue
            ## Detect the beginning of a new message:
            ## two consecutive lines starting with 'From'
            # if line.startswith("From"):
            #    if prev_line_from:
            #        # Replace the first 'From' with 'EmailFrom' in the previous line
            #        # Go back and update the previous line in the file
            #        outfile.seek(outfile.tell() - len(last_line))
            #        outfile.write(last_line.replace("From", "EmailFrom", 1))
            #        outfile.truncate()
            #        outfile.write(line)
            #        prev_line_from = True
            #        last_line = line
            #        continue
            #    prev_line_from = True
            # else:
            #    prev_line_from = False
            outfile.write(line)
            # last_line = line
    try:
        os.close(temp_fd)
    except Exception:
        pass

    try:
        if search_type == "sparse":
            # Handle sparse search with BM25
            ProcessFileSparse(cleaned_temp_file, file_path)
        elif search_type == "dense":
            # Handle dense search with ChromaDB
            ProcessFileDense(cleaned_temp_file, file_path, compute_location)
        else:
            raise ValueError(f"Unsupported search type: {search_type}")
    finally:
        # Clean up the temporary file
        try:
            os.remove(cleaned_temp_file)
        except Exception:
            pass


def ProcessFileDense(cleaned_temp_file, file_path, compute_location):
    """
    Process file for dense vector search using ChromaDB
    """
    # Get a retriever instance
    retriever = BuildRetriever("dense", compute_location)
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
        separators=["\n\nFrom"], chunk_size=1, chunk_overlap=0
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
        db_dir = GetRetrieverParam("db_dir")
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
