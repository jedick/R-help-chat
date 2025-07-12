from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langgraph.graph import START, END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import List, Annotated, TypedDict
from transformers import AutoTokenizer
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
from bm25s_retriever import BM25SRetriever
from build_retriever import BuildRetriever, GetRetrieverParam
from process_file import ProcessFile

# R-help-chat
# First version by Jeffrey Dick on 2025-06-29

# Embedding API (remote or local)
embedding_type = "remote"
# Chat API (remote or remote)
chat_type = "remote"

# Suppress these messages:
# INFO:httpx:HTTP Request: POST https://api.openai.com/v1/embeddings "HTTP/1.1 200 OK"
# INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
# https://community.openai.com/t/suppress-http-request-post-message/583334/8
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)

# Create a prompt template
prompt = ChatPromptTemplate.from_template(
    """Use only the provided context to answer the following question.
    If the context does not have enough information to answer the question, say so.
    Context: {context}
    Question: {question}
    """
)


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
        ProcessFile(file_path, "sparse", embedding_type)
        print(f"Processed {file_path} for sparse search")

    # Get a dense retriever instance
    retriever = BuildRetriever("dense", embedding_type)
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
                retriever.vectorstore.delete(results["ids"])
                update_file = True

        if add_file:
            ProcessFile(file_path, "dense", embedding_type)

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


def GetChatModel(chat_type):
    """
    Get a chat model.

    Args:
        chat_type: Type of chat API (remote or local)
    """

    if chat_type == "remote":
        chat_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    if chat_type == "local":
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

    chat_model = GetChatModel(chat_type)

    # Building an LCEL retrieval chain
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | chat_model
        | StrOutputParser()
    )

    # Invoking the retrieval chain
    result = chain.invoke(query)
    return result


def RunGraph(query, search_type: str = "hybrid_rr", chat_type=chat_type):
    """
    Run graph for retrieval and chat, with source citations

    Args:
        search_type: Type of search to use. Options: "dense", "sparse", "sparse_rr", "hybrid", "hybrid_rr"
        chat_type: Type of chat API (remote or local)

    Example: RunGraph("What R functions are discussed?")
    """

    # For tracing
    # os.environ["LANGSMITH_TRACING"] = "true"
    # os.environ["LANGSMITH_PROJECT"] = "R-help-chat"
    # For LANGCHAIN_API_KEY
    # load_dotenv(dotenv_path=".env", override=True)

    chat_model = GetChatModel(chat_type)

    # Desired schema for response
    class ResponseWithSources(TypedDict):
        """A response to the question, with sources."""

        answer: str
        sources: Annotated[
            List[str],
            ...,
            "List of sources (sender's name and date in From: email headers) used to answer the question",
        ]

    # Define state for application
    class State(TypedDict):
        question: str
        context: List[Document]
        response: ResponseWithSources

    # Define retrieval step
    def retrieve(state: State):
        # Get retriever instance
        retriever = BuildRetriever(search_type, embedding_type)
        if retriever is None:
            raise Exception(
                "No retriever available. Please process some documents first."
            )
        retrieved_docs = retriever.invoke(state["question"])
        return {"context": retrieved_docs}

    # Define generation step
    def generate(state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = prompt.invoke(
            {"question": state["question"], "context": docs_content}
        )
        structured_chat_model = chat_model.with_structured_output(ResponseWithSources)
        response = structured_chat_model.invoke(messages)
        return {"response": response}

    # Compile application
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()

    # Because we're tracking the retrieved context in our application's state, it is accessible after invoking the application:
    # print(f'Context: {result["context"]}\n\n')
    # print(f'Response: {result["response"]}')
    result = graph.invoke({"question": query})
    return result["response"]


def BuildChatGraph(search_type: str = "hybrid_rr", chat_type=chat_type):
    """
    Build graph for chat (conversational RAG with memory)

    Args:
        search_type: Type of search to use. Options: "dense", "sparse", "sparse_rr", "hybrid", "hybrid_rr"
        chat_type: Type of chat API (remote or local)

    Example: RunGraph("What R functions are discussed?")

    Based on:
        https://python.langchain.com/docs/tutorials/qa_chat_history
        https://python.langchain.com/docs/how_to/qa_sources
    """

    # For tracing
    # os.environ["LANGSMITH_TRACING"] = "true"
    # os.environ["LANGSMITH_PROJECT"] = "R-help-chat"
    # For LANGCHAIN_API_KEY
    # load_dotenv(dotenv_path=".env", override=True)

    # Get chat model, used in both respond_or_retrieve and generate
    chat_model = GetChatModel(chat_type)

    # Define start of system message, used in both respond_or_retrieve and generate
    system_message_prefix = (
        "You are an assistant that answers questions about R programming. "
        "Do not respond with your own knowledge or ask the user for more information. "
        "Instead, use a tool to retrieve information related to the query from the R-help mailing list archives. "
        "Use only the retrieved information to provide a helpful answer. "
        "If the retrieved information is insufficient to answer the question, say so. "
    )

    # Define retrieval tool
    # We propagate the retrieved documents as artifacts on the tool messages.
    # That makes it easy to pluck out the retrieved documents.
    # Below, we add them as an additional key in the state, for convenience.
    # Define the response format of the tool as "content_and_artifact":
    @tool(response_format="content_and_artifact")
    def retrieve(query: str):
        """Retrieve information related to a query from the R-help mailing list archives"""
        # Get retriever instance
        retriever = BuildRetriever(search_type, embedding_type)
        retrieved_docs = retriever.invoke(query)
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs

    # Define state for application
    # Represent the state of our RAG application using a sequence of messages to enable:
    #   - Tool-calling features of chat models
    #   - A "back-and-forth" conversational user experience
    # We will have:
    #   - User input as a HumanMessage
    #   - Vector store query as an AIMessage with tool calls
    #   - Retrieved documents as a ToolMessage.
    #   - Final response as a AIMessage
    # Leveraging tool-calling to interact with a retrieval step allows a model to rewrite user queries into more effective search queries
    class ChatMessagesState(MessagesState):
        # Add a context key to the state to store retrieved documents
        context: List[Document]

    # Define response or retrieval step (entry point)
    # NOTE: This has to be ChatMessagesState, not MessagesState, to access step["context"]
    def respond_or_retrieve(state: ChatMessagesState):
        """Generate AI response or tool call for retrieval"""
        chat_model_with_tools = chat_model.bind_tools([retrieve])
        # response = chat_model_with_tools.invoke(state["messages"])
        response = chat_model_with_tools.invoke(
            [SystemMessage(system_message_prefix)] + state["messages"]
        )
        # MessagesState appends messages to state instead of overwriting
        return {"messages": [response]}

    # Define retrieval step
    tools = ToolNode([retrieve])

    # Define generation step
    def generate(state: MessagesState):
        """Generate a response using the retrieved content"""
        # Get generated ToolMessages
        recent_tool_messages = []
        for message in reversed(state["messages"]):
            if message.type == "tool":
                recent_tool_messages.append(message)
            else:
                break
        tool_messages = recent_tool_messages[::-1]

        # Format into prompt
        docs_content = "\n\n".join(doc.content for doc in tool_messages)
        system_message_with_context = (
            f"{system_message_prefix}"
            "\n\n"
            "### Retrieved Information:"
            f"{docs_content}"
        )
        conversation_messages = [
            message
            for message in state["messages"]
            if message.type in ("human", "system")
            or (message.type == "ai" and not message.tool_calls)
        ]
        prompt = [SystemMessage(system_message_with_context)] + conversation_messages

        # Run the model
        response = chat_model.invoke(prompt)
        context = []
        # Pluck out the retrieved documents and populate them in the state
        for tool_message in tool_messages:
            context.extend(tool_message.artifact)
        return {"messages": [response], "context": context}

    # Initialize a graph object
    graph_builder = StateGraph(MessagesState)
    # Add nodes
    # A node that fields the user input, either responding directly or using a tool
    graph_builder.add_node(respond_or_retrieve)
    # A node for the retriever tool that executes the retrieval step
    graph_builder.add_node(tools)
    # A node that generates the final response using the retrieved context
    graph_builder.add_node(generate)
    # Set entry point
    graph_builder.set_entry_point("respond_or_retrieve")
    # Add edges
    graph_builder.add_conditional_edges(
        "respond_or_retrieve",
        tools_condition,
        # The first respond_or_retrieve step can respond directly to the user or generate a tool call
        {END: END, "tools": "tools"},
    )
    graph_builder.add_edge("tools", "generate")
    graph_builder.add_edge("generate", END)

    return graph_builder


def RunChat(query: str, thread_id=None):
    """Run chat with graph for conversational RAG

    Args:
        query: User query to start the chat

    Example;
        RunChat("Help with parsing REST API response.")
    """

    # Build the graph
    graph_builder = BuildChatGraph()

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
        step["messages"][-1].pretty_print()

    # We see that the retrieved Document objects are accessible from the application state.
    # step["context"]
    return step
