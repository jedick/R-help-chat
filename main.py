from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from datetime import datetime
from transformers import AutoTokenizer
import os
import glob
import torch

# To use OpenAI models (remote)
from langchain_openai import ChatOpenAI

# To use Hugging Face models (local)
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

# Local modules
from bm25s_retriever import BM25SRetriever
from build_retriever import BuildRetriever, GetRetrieverParam
from process_file import ProcessFile
from util import SuppressStderr

# R-help-chat
# First version by Jeffrey Dick on 2025-06-29

# Embedding API (remote or local)
embedding_api = "remote"
# LLM API (remote or remote)
llm_api = "remote"


def ProcessDirectory(path):
    """
    Update vector store and sparse index for files in a directory, only adding new or updated files
    "path": directory to process

    Usage example:
    ProcessDirectory("R-help")
    """

    # TODO: use UUID to process only changed documents
    # https://stackoverflow.com/questions/76265631/chromadb-add-single-document-only-if-it-doesnt-exist

    # BM25 doesn't have the same metadata tracking as ChromaDB
    # For now, we'll process all files when using sparse search
    file_paths = glob.glob(f"{path}/*.txt")
    for file_path in file_paths:
        ProcessFile(file_path, "sparse", embedding_api)
        print(f"Processed {file_path} for sparse search")

    # Get a dense retriever instance
    with SuppressStderr():
        retriever = BuildRetriever("dense", embedding_api)
    # List all text files in target directory
    file_paths = glob.glob(f"{path}/*.txt")
    # Loop over files
    for file_path in file_paths:
        # Look for existing embeddings for this file
        with SuppressStderr():
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
                with SuppressStderr():
                    retriever.vectorstore.delete(results["ids"])
                update_file = True

        if add_file:
            ProcessFile(file_path, "dense", embedding_api)

        if update_file:
            print(f"Updated embeddings for {file_path}")
            # Clear out the unused parent files
            # The used doc_ids are the files to keep
            used_doc_ids = [
                d["doc_id"] for d in retriever.vectorstore.get()["metadatas"]
            ]
            files_to_keep = list(set(used_doc_ids))
            # Get all files in the file store
            file_store = GetRetrieverParam("file_store")
            all_files = os.listdir(file_store)
            # Iterate through the files and delete those not in the list
            for file in all_files:
                if file not in files_to_keep:
                    file_path = os.path.join(file_store, file)
                    os.remove(file_path)
        elif add_file:
            print(f"Added embeddings for {file_path}")
        else:
            print(f"No change for {file_path}")


def ListDocuments():

    with SuppressStderr():
        # Get retriever instance
        retriever = BuildRetriever("dense", embedding_api)
        # Retrieve all document IDs
        document_ids = retriever.vectorstore.get()["ids"]
    # Return the document IDs
    return document_ids


def QueryDatabase(query, search_type: str = "hybrid", llm_api = llm_api):
    """
    Retrieve documents from database and query with LLM

    Example: QueryDatabase("What R functions are discussed?")
    """

    # Get retriever instance
    with SuppressStderr():
        retriever = BuildRetriever(search_type, embedding_api)

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
    ## Define a model
    if llm_api == "remote":
        chat_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    if llm_api == "local":
        llm = HuggingFacePipeline.from_model_id(
            model_id="google/gemma-3-4b-it",
            task="text-generation",
            pipeline_kwargs=dict(
                max_new_tokens=512,
                do_sample=False,
                # Without this output begins repeating
                repetition_penalty=1.1,
                return_full_text=False,
            ),
            # We need this to load the model in BF16 instead of fp32 (torch.float)
            model_kwargs={"torch_dtype": torch.bfloat16},
        )
        # Need to provide the tokenizer here, or get OSError:
        # None is not a local folder and is not a valid model identifier listed on 'https://huggingface.co/models'
        # https://github.com/langchain-ai/langchain/issues/31324
        tokenizer = AutoTokenizer.from_pretrained(llm.model_id)
        chat_model = ChatHuggingFace(llm=llm, tokenizer=tokenizer)

    # Building an LCEL retrieval chain
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | chat_model
        | StrOutputParser()
    )

    # Invoking the retrieval chain
    with SuppressStderr():
        result = chain.invoke(query)
    return result
