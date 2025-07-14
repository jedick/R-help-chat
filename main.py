from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langgraph.checkpoint.memory import MemorySaver
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from dotenv import load_dotenv
from datetime import datetime
import os
import glob
import torch
import logging

# To use OpenAI models (remote)
from langchain_openai import ChatOpenAI

# To use Hugging Face models (local)
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

# Local modules
from process_file import ProcessFile
from build_retriever import BuildRetriever, GetRetrieverParam
from build_graph import BuildGraph

# R-help-chat
# First version by Jeffrey Dick on 2025-06-29

# Embedding API (remote or local)
embedding_type = "remote"
# Chat API (remote or remote)
chat_type = "remote"

# Don't try to use local models without a GPU
if not torch.cuda.is_available() and (
    embedding_type == "local" or chat_type == "local"
):
    raise Exception("Local model selected without GPU")

# Suppress these messages:
# INFO:httpx:HTTP Request: POST https://api.openai.com/v1/embeddings "HTTP/1.1 200 OK"
# INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
# https://community.openai.com/t/suppress-http-request-post-message/583334/8
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)


def ProcessDirectory(path):
    """
    Update vector store and sparse index for files in a directory, only adding new or updated files
    "path": directory to process

    Usage example:
    ProcessDirectory("R-help")
    """

    # TODO: use UUID to process only changed documents
    # https://stackoverflow.com/questions/76265631/chromadb-add-single-document-only-if-it-doesnt-exist

    # Get a dense retriever instance
    retriever = BuildRetriever("dense", embedding_type)

    # List all text files in target directory
    file_paths = glob.glob(f"{path}/*.txt")
    for file_path in file_paths:

        # Process file for sparse search (BM25S)
        ProcessFile(file_path, "sparse", embedding_type)

        # Logic for dense search: skip file if already indexed
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
                retriever.vectorstore.delete(results["ids"])
                update_file = True

        if add_file:
            ProcessFile(file_path, "dense", embedding_type)

        if update_file:
            print(f"Chroma: updated embeddings for {file_path}")
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
            print(f"Chroma: added embeddings for {file_path}")
        else:
            print(f"Chroma: no change for {file_path}")


def GetChatModel(chat_type):
    """
    Get a chat model.

    Args:
        chat_type: Type of chat API (remote or local)
    """

    if chat_type == "remote":

        chat_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    if chat_type == "local":

        # Define the pipeline to pass to the HuggingFacePipeline class
        # https://huggingface.co/blog/langchain
        model_id = "HuggingFaceTB/SmolLM3-3B"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            # We need this to load the model in BF16 instead of fp32 (torch.float)
            torch_dtype=torch.bfloat16,
        )

        # ToolCallingLLM needs return_full_text=False in order to parse just the assistant response;
        # the JSON function descriptions in the full response cause an error in ToolCallingLLM
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            return_full_text=False,
            # It seems that max_new_tokens has to be specified here, not in .invoke()
            max_new_tokens=1000,
        )

        llm = HuggingFacePipeline(pipeline=pipe)
        chat_model = ChatHuggingFace(llm=llm)

    return chat_model


def RunChain(query, search_type: str = "hybrid_rr", chat_type=chat_type):
    """
    Run chain to retrieve documents and send to chat

    Args:
        search_type: Type of search to use. Options: "dense", "sparse", "sparse_rr", "hybrid", "hybrid_rr"
        chat_type: Type of chat API (remote or local)

    Example: RunChain("What R functions are discussed?")
    """

    # Get retriever instance
    retriever = BuildRetriever(search_type, embedding_type)

    if retriever is None:
        return "No retriever available. Please process some documents first."

    # Get chat model (LLM)
    chat_model = GetChatModel(chat_type)

    # Create a prompt template
    prompt = ChatPromptTemplate.from_template(
        """Use only the provided context to answer the following question.
        If the context does not have enough information to answer the question, say so.
        Context: {context}
        Question: {question}
        """
    )

    # Build an LCEL retrieval chain
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | chat_model
        | StrOutputParser()
    )

    # Invoke the retrieval chain
    result = chain.invoke(query)
    return result


def RunGraph(
    query: str,
    search_type: str = "hybrid_rr",
    chat_type=chat_type,
    think_retrieve=False,
    think_generate=False,
    thread_id=None,
):
    """Run graph for conversational RAG app

    Args:
        query: User query to start the chat
        search_type: Type of search to use. Options: "dense", "sparse", "sparse_rr", "hybrid", "hybrid_rr"
        chat_type: Type of chat API (remote or local)
        think_retrieve: Whether to use thinking mode for retrieval (tool-calling)
        think_generate: Whether to use thinking mode for generation
        thread_id: Thread ID for memory (optional)

    Example:
        RunGraph("Help with parsing REST API response.")
    """

    # Get retriever instance
    retriever = BuildRetriever(search_type, embedding_type)
    # Get chat model used in both respond_or_retrieve and generate steps
    chat_model = GetChatModel(chat_type)
    # Build the graph
    graph_builder = BuildGraph(
        retriever=retriever,
        chat_model=chat_model,
        think_retrieve=think_retrieve,
        think_generate=think_generate,
    )

    # FIXME: Use thread id for memory if given
    # TypeError: Type is not msgpack serializable: ToolMessage
    # https://github.com/langchain-ai/langgraph/issues/5054
    # https://github.com/langchain-ai/langgraph/pull/5115
    if thread_id is None:
        graph = graph_builder.compile()
        config = None
    else:
        # FIXME: TypeError: Type is not msgpack serializable: ToolMessage
        # https://github.com/langchain-ai/langgraph/issues/5054
        # https://github.com/langchain-ai/langgraph/pull/5115
        # Compile our application with an in-memory checkpointer
        memory = MemorySaver()
        graph = graph_builder.compile(checkpointer=memory)
        # Specify an ID for the thread
        config = {"configurable": {"thread_id": thread_id}}

    # When executing a search, we can stream the steps to observe the query generation, retrieval, and answer generation:
    for step in graph.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="values",
        config=config,
    ):
        if not step["messages"][-1].type == "tool":
            step["messages"][-1].pretty_print()

    # To get the last message content: step["messages"][-1].content
    # To get the retrieved context and cited sources:
    # step["context"]
    # step["sources"]
    return step
