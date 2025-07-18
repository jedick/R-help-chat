from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import ToolMessage
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from datetime import datetime
import os
import glob
import torch
import logging
import ast

# To use OpenAI models (cloud)
from langchain_openai import ChatOpenAI

# To use Hugging Face models (edge)
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

# Local modules
from index import ProcessFile
from retriever import BuildRetriever, GetRetrieverParam
from graph import BuildGraph
from prompts import generate_prompt

# R-help-chat
# First version by Jeffrey Dick on 2025-06-29

# Suppress these messages:
# INFO:httpx:HTTP Request: POST https://api.openai.com/v1/embeddings "HTTP/1.1 200 OK"
# INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
# https://community.openai.com/t/suppress-http-request-post-message/583334/8
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)


def ProcessDirectory(path, compute_location):
    """
    Update vector store and sparse index for files in a directory, only adding new or updated files

    Args:
        path: Directory to process
        compute_location: Compute location for embeddings (cloud or edge)

    Usage example:
        ProcessDirectory("R-help", "cloud")
    """

    # TODO: use UUID to process only changed documents
    # https://stackoverflow.com/questions/76265631/chromadb-add-single-document-only-if-it-doesnt-exist

    # Get a dense retriever instance
    retriever = BuildRetriever(compute_location, "dense")

    # List all text files in target directory
    file_paths = glob.glob(f"{path}/*.txt")
    for file_path in file_paths:

        # Process file for sparse search (BM25S)
        ProcessFile(file_path, "sparse", compute_location)

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
            ProcessFile(file_path, "dense", compute_location)

        if update_file:
            print(f"Chroma: updated embeddings for {file_path}")
            # Clear out the unused parent files
            # The used doc_ids are the files to keep
            used_doc_ids = [
                d["doc_id"] for d in retriever.vectorstore.get()["metadatas"]
            ]
            files_to_keep = list(set(used_doc_ids))
            # Get all files in the file store
            db_dir = GetRetrieverParam("db_dir")
            file_store = f"{db_dir}/file_store_{compute_location}"
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


def GetChatModel(compute_location):
    """
    Get a chat model.

    Args:
        compute_location: Compute location for chat model (cloud or edge)
    """

    if compute_location == "cloud":

        chat_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    if compute_location == "edge":

        # Don't try to use edge models without a GPU
        if compute_location == "edge" and not torch.cuda.is_available():
            raise Exception("Edge chat model selected without GPU")

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


def RunChain(
    query,
    compute_location: str = "cloud",
    search_type: str = "hybrid",
    think: bool = False,
):
    """
    Run chain to retrieve documents and send to chat

    Args:
        query: User's query
        compute_location: Compute location for embedding and chat models (cloud or edge)
        search_type: Type of search to use. Options: "dense", "sparse", or "hybrid"
        think: Control thinking mode for SmolLM3

    Example:
        RunChain("What R functions are discussed?")
    """

    # Get retriever instance
    retriever = BuildRetriever(compute_location, search_type)

    if retriever is None:
        return "No retriever available. Please process some documents first."

    # Get chat model (LLM)
    chat_model = GetChatModel(compute_location)

    # Control thinking for SmolLM3
    system_prompt = generate_prompt
    if hasattr(chat_model, "model_id") and not think:
        system_prompt = f"/no_think\n{system_prompt}"

    # Create a prompt template
    system_template = ChatPromptTemplate.from_messages([SystemMessage(system_prompt)])
    human_template = ChatPromptTemplate.from_template(
        """"
        ### Question:

        {question}

        ### Retrieved Emails:

        {context}
        """
    )
    prompt_template = system_template + human_template

    # Build an LCEL retrieval chain
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt_template
        | chat_model
        | StrOutputParser()
    )

    # Invoke the retrieval chain
    result = chain.invoke(query)
    return result


def GetGraphAndConfig(
    compute_location: str = "cloud",
    search_type: str = "hybrid",
    top_k: int = 6,
    think_retrieve=True,
    think_generate=False,
    thread_id=None,
):
    """Get graph for conversational RAG app

    Args:
        compute_location: Compute location for embedding and chat models (cloud or edge)
        search_type: Type of search to use. Options: "dense", "sparse", or "hybrid"
        top_k: Number of documents to retrieve
        think_retrieve: Whether to use thinking mode for retrieval (tool-calling)
        think_generate: Whether to use thinking mode for generation
        thread_id: Thread ID for memory (optional)

    Example:
        RunGraph("Help with parsing REST API response.")
    """

    # Get chat model used in both respond_or_retrieve and generate steps
    chat_model = GetChatModel(compute_location)
    # Build the graph
    graph_builder = BuildGraph(
        chat_model,
        compute_location,
        search_type,
        top_k,
        think_retrieve,
        think_generate,
    )

    if thread_id is None:
        graph = graph_builder.compile()
        config = None
    else:
        # Compile our application with an in-memory checkpointer
        memory = MemorySaver()
        graph = graph_builder.compile(checkpointer=memory)
        # Specify an ID for the thread
        config = {"configurable": {"thread_id": thread_id}}

    return graph, config


def RunGraph(
    query: str,
    compute_location: str = "cloud",
    search_type: str = "hybrid",
    top_k: int = 6,
    **kwargs,
):
    """Run graph for conversational RAG app

    Args:
        query: User query to start the chat
        compute_location: Compute location for embedding and chat models (cloud or edge)
        search_type: Type of search to use. Options: "dense", "sparse", or "hybrid"
        top_k: Number of documents to retrieve
        **kwargs: Additional keyword arguments for GetGraphAndConfig()

    Example:
        RunGraph("Help with parsing REST API response.")
    """

    # Get graph and config
    graph, config = GetGraphAndConfig(compute_location, search_type, **kwargs)

    # Stream the steps to observe the query generation, retrieval, and answer generation:
    #   - User input as a HumanMessage
    #   - Vector store query as an AIMessage with tool calls
    #   - Retrieved documents as a ToolMessage.
    #   - Final response as a AIMessage
    for state in graph.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="values",
        config=config,
    ):
        if not state["messages"][-1].type == "tool":
            state["messages"][-1].pretty_print()

    # Parse the state to return the results
    try:
        answer, citations = ast.literal_eval(state["messages"][-1].content)
    except:
        # In case we get an answer without citations
        answer = state["messages"][-1].content
        citations = None
    result = {"answer": answer}
    if citations:
        result["citations"] = citations
    # Parse tool messages to get context
    tool_messages = [msg for msg in state["messages"] if type(msg) == ToolMessage]
    # Get content from the most recent retrieve_emails response
    content = None
    for msg in tool_messages:
        if msg.name == "retrieve_emails":
            content = msg.content
    # Parse it into a list of emails
    if content:
        retrieved_emails = content.replace("### Retrieved Emails:\n\n\n\n", "").split(
            "--- --- --- --- Next Email --- --- --- ---\n\n"
        )
        result["retrieved_emails"] = retrieved_emails

    return result
