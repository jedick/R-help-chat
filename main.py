from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage
from langchain_core.messages import ToolMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from datetime import datetime
import logging
import glob
import ast
import os

# Local modules
from retriever import BuildRetriever
from prompts import answer_prompt
from index import ProcessFile
from graph import BuildGraph

# -----------
# R-help-chat
# -----------
# First version by Jeffrey Dick on 2025-06-29
# Updated to LangChain v1 and Chroma 6 on 2026-01-03

# Setup environment variables
load_dotenv(dotenv_path=".env", override=True)

# Define the OpenAI model
openai_model = "gpt-4o-mini"

# Suppress these messages:
# INFO:httpx:HTTP Request: POST https://api.openai.com/v1/embeddings "HTTP/1.1 200 OK"
# INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
# https://community.openai.com/t/suppress-http-request-post-message/583334/8
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)


def ProcessCollection(email_dir, db_dir):
    """
    Update vector store and sparse index for files in a directory, only adding new or updated files

    Args:
        email_dir: Email directory to process
        db_dir: Database directory

    Usage example:
        ProcessCollection("R-help", "db")
    """

    # TODO: use UUID to process only changed documents
    # https://stackoverflow.com/questions/76265631/chromadb-add-single-document-only-if-it-doesnt-exist

    # Get last part of path
    # https://stackoverflow.com/questions/3925096/how-to-get-only-the-last-part-of-a-path-in-python
    collection = os.path.basename(os.path.normpath(email_dir))
    # Get a dense retriever instance
    retriever = BuildRetriever(db_dir, collection, "dense")

    # List all text files in target directory
    file_paths = glob.glob(f"{email_dir}/*.txt")
    for file_path in file_paths:

        # Process file for sparse search (BM25S)
        ProcessFile(file_path, db_dir, collection, "sparse")

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
            ProcessFile(file_path, db_dir, collection, "dense")

        if update_file:
            print(f"Chroma: updated embeddings for {file_path}")
            # Clear out the unused parent files
            # The used doc_ids are the files to keep
            used_doc_ids = [
                d["doc_id"] for d in retriever.vectorstore.get()["metadatas"]
            ]
            files_to_keep = list(set(used_doc_ids))
            # Get all files in the file store
            file_store = os.path.join(db_dir, collection, "file_store")
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


def RunChain(
    query: str,
    db_dir: str,
    collection: str,
    search_type: str = "hybrid",
):
    """
    Run chain to retrieve documents and send to chat

    Args:
        query: User's query
        db_dir: Database directory
        collection: Email collection
        search_type: Type of search to use. Options: "dense", "sparse", or "hybrid"

    Example:
        RunChain("What R functions are discussed?", "db", "R-help")
    """

    # Get retriever instance
    retriever = BuildRetriever(db_dir, collection, search_type)

    if retriever is None:
        return "No retriever available. Please process some documents first."

    # Get chat model (LLM)
    chat_model = ChatOpenAI(model=openai_model, temperature=0)

    # Get system prompt
    system_prompt = answer_prompt()

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
    db_dir: str,
    collection: str,
    search_type: str = "hybrid",
    top_k: int = 6,
    thread_id=None,
):
    """Run graph for conversational RAG app

    Args:
        query: User query to start the chat
        db_dir: Database directory
        collection: Email collection
        search_type: Type of search to use. Options: "dense", "sparse", or "hybrid"
        top_k: Number of documents to retrieve
        thread_id: Thread ID for memory (optional)

    Example:
        RunGraph("Help with parsing REST API response.")
    """

    # Get chat model used in both query and generate steps
    chat_model = ChatOpenAI(model=openai_model, temperature=0)
    # Build the graph
    graph_builder = BuildGraph(
        chat_model,
        db_dir,
        collection,
        search_type,
        top_k,
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
