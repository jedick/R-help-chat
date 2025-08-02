from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage
from langchain_core.messages import ToolMessage
from dotenv import load_dotenv
from datetime import datetime
import logging
import torch
import glob
import ast
import os

# Imports for local and remote chat models
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_openai import ChatOpenAI

# Local modules
from pipeline import MyTextGenerationPipeline
from retriever import BuildRetriever, db_dir
from prompts import answer_prompt
from index import ProcessFile
from graph import BuildGraph

# -----------
# R-help-chat
# -----------
# First version by Jeffrey Dick on 2025-06-29

# Setup environment variables
load_dotenv(dotenv_path=".env", override=True)

# Define the remote (OpenAI) model
openai_model = "gpt-4o-mini"

# Get the local model ID
model_id = os.getenv("MODEL_ID")
if model_id is None:
    # model_id = "HuggingFaceTB/SmolLM3-3B"
    model_id = "google/gemma-3-12b-it"
    # model_id = "Qwen/Qwen3-14B"

# Suppress these messages:
# INFO:httpx:HTTP Request: POST https://api.openai.com/v1/embeddings "HTTP/1.1 200 OK"
# INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
# https://community.openai.com/t/suppress-http-request-post-message/583334/8
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)


def ProcessDirectory(path, compute_mode):
    """
    Update vector store and sparse index for files in a directory, only adding new or updated files

    Args:
        path: Directory to process
        compute_mode: Compute mode for embeddings (remote or local)

    Usage example:
        ProcessDirectory("R-help", "remote")
    """

    # TODO: use UUID to process only changed documents
    # https://stackoverflow.com/questions/76265631/chromadb-add-single-document-only-if-it-doesnt-exist

    # Get a dense retriever instance
    retriever = BuildRetriever(compute_mode, "dense")

    # List all text files in target directory
    file_paths = glob.glob(f"{path}/*.txt")
    for file_path in file_paths:

        # Process file for sparse search (BM25S)
        ProcessFile(file_path, "sparse", compute_mode)

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
            ProcessFile(file_path, "dense", compute_mode)

        if update_file:
            print(f"Chroma: updated embeddings for {file_path}")
            # Clear out the unused parent files
            # The used doc_ids are the files to keep
            used_doc_ids = [
                d["doc_id"] for d in retriever.vectorstore.get()["metadatas"]
            ]
            files_to_keep = list(set(used_doc_ids))
            # Get all files in the file store
            file_store = f"{db_dir}/file_store_{compute_mode}"
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


def GetChatModel(compute_mode, ckpt_dir=None):
    """
    Get a chat model.

    Args:
        compute_mode: Compute mode for chat model (remote or local)
        ckpt_dir: Checkpoint directory for model weights (optional)
    """

    if compute_mode == "remote":

        chat_model = ChatOpenAI(model=openai_model, temperature=0)

    if compute_mode == "local":

        # Don't try to use local models without a GPU
        if compute_mode == "local" and not torch.cuda.is_available():
            raise Exception("Local chat model selected without GPU")

        # Define the pipeline to pass to the HuggingFacePipeline class
        # https://huggingface.co/blog/langchain
        id_or_dir = ckpt_dir if ckpt_dir else model_id
        tokenizer = AutoTokenizer.from_pretrained(id_or_dir)
        model = AutoModelForCausalLM.from_pretrained(
            id_or_dir,
            # We need this to load the model in BF16 instead of fp32 (torch.float)
            torch_dtype=torch.bfloat16,
            # Enable FlashAttention (requires pip install flash-attn)
            # https://huggingface.co/docs/transformers/en/attention_interface
            # https://huggingface.co/docs/transformers/perf_infer_gpu_one#flashattention
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        # For Flash Attention version of Qwen3
        tokenizer.padding_side = "left"

        # Use MyTextGenerationPipeline with custom preprocess() method
        pipe = MyTextGenerationPipeline(
            model=model,
            tokenizer=tokenizer,
            # ToolCallingLLM needs return_full_text=False in order to parse just the assistant response
            return_full_text=False,
            # It seems that max_new_tokens has to be specified here, not in .invoke()
            max_new_tokens=2000,
            # Use padding for proper alignment for FlashAttention
            # Part of fix for: "RuntimeError: p.attn_bias_ptr is not correctly aligned"
            # https://github.com/google-deepmind/gemma/issues/169
            padding="longest",
        )
        # We need the task so HuggingFacePipeline can deal with our class
        pipe.task = "text-generation"

        llm = HuggingFacePipeline(pipeline=pipe)
        chat_model = ChatHuggingFace(llm=llm)

    return chat_model


def RunChain(
    query,
    compute_mode: str = "remote",
    search_type: str = "hybrid",
    think: bool = False,
):
    """
    Run chain to retrieve documents and send to chat

    Args:
        query: User's query
        compute_mode: Compute mode for embedding and chat models (remote or local)
        search_type: Type of search to use. Options: "dense", "sparse", or "hybrid"
        think: Control thinking mode for SmolLM3

    Example:
        RunChain("What R functions are discussed?")
    """

    # Get retriever instance
    retriever = BuildRetriever(compute_mode, search_type)

    if retriever is None:
        return "No retriever available. Please process some documents first."

    # Get chat model (LLM)
    chat_model = GetChatModel(compute_mode)

    # Get prompt with /no_think for SmolLM3/Qwen
    system_prompt = answer_prompt(chat_model)

    # Create a prompt template
    system_template = ChatPromptTemplate.from_messages([SystemMessage(system_prompt)])
    # NOTE: Each new email starts with \n\n\nFrom, so we don't need newlines after Retrieved Emails:
    human_template = ChatPromptTemplate.from_template(
        """"
        ### Question:

        {question}

        ### Retrieved Emails:{context}
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


def RunGraph(
    query: str,
    compute_mode: str = "remote",
    search_type: str = "hybrid",
    top_k: int = 6,
    think_query=False,
    think_answer=False,
    thread_id=None,
):
    """Run graph for conversational RAG app

    Args:
        query: User query to start the chat
        compute_mode: Compute mode for embedding and chat models (remote or local)
        search_type: Type of search to use. Options: "dense", "sparse", or "hybrid"
        top_k: Number of documents to retrieve
        think_query: Whether to use thinking mode for the query
        think_answer: Whether to use thinking mode for the answer
        thread_id: Thread ID for memory (optional)

    Example:
        RunGraph("Help with parsing REST API response.")
    """

    # Get chat model used in both query and generate steps
    chat_model = GetChatModel(compute_mode)
    # Build the graph
    graph_builder = BuildGraph(
        chat_model,
        compute_mode,
        search_type,
        top_k,
        think_query,
        think_answer,
    )

    # Compile the graph with an in-memory checkpointer
    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)
    # Specify an ID for the thread
    config = {"configurable": {"thread_id": thread_id}}

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

    # Parse the messages for the answer and citations
    try:
        answer, citations = ast.literal_eval(state["messages"][-1].content)
    except:
        # In case we got an answer without citations
        answer = state["messages"][-1].content
        citations = None
    result = {"answer": answer}
    if citations:
        result["citations"] = citations
    # Parse tool messages to get retrieved emails
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
