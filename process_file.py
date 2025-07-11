import tempfile
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# Local modules
from bm25s_retriever import BM25SRetriever
from build_retriever import BuildRetriever, GetRetrieverParam, AddTimestamps


def ProcessFile(file_path, search_type: str = "dense", embedding_type: str = "local"):
    """
    Wrapper function to process file for dense or sparse search

    Args:
        file_path: file to process
        search_type: Type of search to use. Options: "dense", "sparse"
        embedding_type: Type of embedding API (remote or local)
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
            ProcessFileDense(cleaned_temp_file, file_path, embedding_type)
        else:
            raise ValueError(f"Unsupported search type: f{search_type}")
    finally:
        # Clean up the temporary file
        try:
            os.remove(cleaned_temp_file)
        except Exception:
            pass


def ProcessFileDense(cleaned_temp_file, file_path, embedding_type):
    """
    Process file for dense vector search using ChromaDB
    """
    # Get a retriever instance
    retriever = BuildRetriever("dense", embedding_type)
    # Load cleaned text file to document
    loader = TextLoader(cleaned_temp_file)
    document = loader.load()
    # Add documents to vectorstore
    retriever.add_documents(document)
    # Add file timestamps to metadata
    AddTimestamps(file_path)


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

    from langchain.text_splitter import TextSplitter

    class CustomTextSplitter(TextSplitter):
        def split_text(self, text):
            # Custom logic to split on every separator
            return text.split("EmailFrom")

    splitter = CustomTextSplitter()
    chunks = splitter.split_text(documents[0].page_content)

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
