from langchain_core.messages import SystemMessage, ToolMessage, HumanMessage, AIMessage
from langgraph.graph import START, END, MessagesState, StateGraph
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_huggingface import ChatHuggingFace
from typing import Optional
import datetime
import os

# Local modules
from retriever import BuildRetriever
from prompts import query_prompt, answer_prompt, generic_tools_template
from mods.tool_calling_llm import ToolCallingLLM

# For tracing (disabled)
# os.environ["LANGSMITH_TRACING"] = "true"
# os.environ["LANGSMITH_PROJECT"] = "R-help-chat"


def print_message_summaries(messages, header):
    """Print message types and summaries for debugging"""
    if header:
        print(header)
    for message in messages:
        summary_text = ""
        if type(message) == SystemMessage:
            type_txt = "SystemMessage"
            summary_txt = f"length = {len(message.content)}"
        if type(message) == HumanMessage:
            type_txt = "HumanMessage"
            summary_txt = message.content
        if type(message) == AIMessage:
            type_txt = "AIMessage"
            summary_txt = f"length = {len(message.content)}"
        if type(message) == ToolMessage:
            type_txt = "ToolMessage"
            summary_txt = f"length = {len(message.content)}"
        if hasattr(message, "tool_calls"):
            if len(message.tool_calls) != 1:
                summary_txt = f"{summary_txt} with {len(message.tool_calls)} tool calls"
            else:
                summary_txt = f"{summary_txt} with 1 tool call"
        print(f"{type_txt}: {summary_txt}")


def normalize_messages(messages):
    """Normalize messages to sequence of types expected by chat templates"""
    # Copy the most recent HumanMessage to the end
    # (avoids SmolLM and Qwen ValueError: Last message must be a HumanMessage!)
    if not type(messages[-1]) is HumanMessage:
        for msg in reversed(messages):
            if type(msg) is HumanMessage:
                messages.append(msg)
                break
    # Convert tool output (ToolMessage) to AIMessage
    # (avoids SmolLM and Qwen ValueError: Unknown message type: <class 'langchain_core.messages.tool.ToolMessage'>)
    messages = [
        AIMessage(msg.content) if type(msg) is ToolMessage else msg for msg in messages
    ]
    # Delete tool call (AIMessage)
    # (avoids Gemma TemplateError: Conversation roles must alternate user/assistant/user/assistant/...)
    messages = [
        msg
        for msg in messages
        if not hasattr(msg, "tool_calls")
        or (hasattr(msg, "tool_calls") and not msg.tool_calls)
    ]
    return messages


def ToolifyHF(chat_model, system_message):
    """
    Get a Hugging Face model ready for bind_tools().
    """

    # Combine system prompt and tools template
    tool_system_prompt_template = system_message + generic_tools_template

    class HuggingFaceWithTools(ToolCallingLLM, ChatHuggingFace):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

    chat_model = HuggingFaceWithTools(
        llm=chat_model.llm,
        tool_system_prompt_template=tool_system_prompt_template,
    )

    return chat_model


def BuildGraph(
    chat_model,
    compute_mode,
    search_type,
    top_k=6,
    think_query=False,
    think_answer=False,
    embedding_ckpt_dir=None,
):
    """
    Build conversational RAG graph for email retrieval and answering with citations.

    Args:
        chat_model: LangChain chat model from GetChatModel()
        compute_mode: remote or local (for retriever)
        search_type: dense, sparse, or hybrid (for retriever)
        top_k: number of documents to retrieve
        think_query: Whether to use thinking mode for the query
        think_answer: Whether to use thinking mode for the answer

    Based on:
        https://python.langchain.com/docs/how_to/qa_sources
        https://python.langchain.com/docs/tutorials/qa_chat_history
        https://python.langchain.com/docs/how_to/chatbots_memory/

    Usage Example:

        # Build graph with chat model
        from langchain_openai import ChatOpenAI
        chat_model = ChatOpenAI(model="gpt-4o-mini")
        graph = BuildGraph(chat_model, "remote", "hybrid")

        # Add simple in-memory checkpointer
        from langgraph.checkpoint.memory import MemorySaver
        memory = MemorySaver()
        # Compile app
        app = graph.compile(checkpointer=memory)
        # Draw graph
        # nb. change orientation (TD to LR) in langchain_core/runnables/graph_mermaid.py
        #app.get_graph().draw_mermaid_png(output_file_path="graph.png")

        # Run app
        from langchain_core.messages import HumanMessage
        input = "When was has.HLC mentioned?"
        state = app.invoke(
            {"messages": [HumanMessage(content=input)]},
            config={"configurable": {"thread_id": "1"}},
        )

    """

    @tool(parse_docstring=True)
    def retrieve_emails(
        search_query: str,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
        months: Optional[str] = None,
    ) -> str:
        """
        Retrieve emails related to a search query from the R-help mailing list archives.
        Use optional "start_year" and "end_year" arguments to filter by years.
        Use optional "months" argument to search by month.

        Args:
            search_query: Search query (required)
            months: One or more months (optional)
            start_year: Starting year for emails (optional)
            end_year: Ending year for emails (optional)
        """
        retriever = BuildRetriever(
            compute_mode, search_type, top_k, start_year, end_year, embedding_ckpt_dir
        )
        # For now, just add the months to the search query
        if months:
            search_query = " ".join([search_query, months])
        # If the search query is empty, use the years
        if not search_query:
            search_query = " ".join([search_query, start_year, end_year])
        retrieved_docs = retriever.invoke(search_query)
        serialized = "\n\n--- --- --- --- Next Email --- --- --- ---".join(
            # source key has file names (e.g. R-help/2024-December.txt), useful for retrieval and reporting
            "\n\n" + doc.metadata["source"] + doc.page_content
            for doc in retrieved_docs
        )
        retrieved_emails = (
            "### Retrieved Emails:\n\n" + serialized
            if serialized
            else "### No emails were retrieved"
        )
        return retrieved_emails

    @tool(parse_docstring=True)
    def answer_with_citations(answer: str, citations: str) -> str:
        """
        An answer to the question, with citations of the emails used (senders and dates).

        Args:
            answer: An answer to the question
            citations: Citations of emails used to answer the question, e.g. Jane Doe, 2025-07-04; John Smith, 2020-01-01
        """
        return answer, citations

    # Add tools to the local or remote chat model
    is_local = hasattr(chat_model, "model_id")
    if is_local:
        # For local models (ChatHuggingFace with SmolLM, Gemma, or Qwen)
        query_model = ToolifyHF(
            chat_model, query_prompt(chat_model, think=think_query)
        ).bind_tools([retrieve_emails])
        # Don't use answer_with_citations tool because responses with are sometimes unparseable
        answer_model = chat_model
    else:
        # For remote model (OpenAI API)
        query_model = chat_model.bind_tools([retrieve_emails])
        answer_model = chat_model.bind_tools([answer_with_citations])

    # Initialize the graph object
    graph = StateGraph(MessagesState)

    def query(state: MessagesState):
        """Queries the retriever with the chat model"""
        if is_local:
            # Don't include the system message here because it's defined in ToolCallingLLM
            messages = state["messages"]
            # print_message_summaries(messages, "--- query: before normalization ---")
            messages = normalize_messages(messages)
            # print_message_summaries(messages, "--- query: after normalization ---")
        else:
            messages = [SystemMessage(query_prompt(chat_model))] + state["messages"]
        response = query_model.invoke(messages)

        return {"messages": response}

    def answer(state: MessagesState):
        """Generates an answer with the chat model"""
        if is_local:
            messages = state["messages"]
            # print_message_summaries(messages, "--- answer: before normalization ---")
            messages = normalize_messages(messages)
            # Add the system message here because we're not using tools
            messages = [
                SystemMessage(answer_prompt(chat_model, think=think_answer))
            ] + messages
            # print_message_summaries(messages, "--- answer: after normalization ---")
        else:
            messages = [
                SystemMessage(answer_prompt(chat_model, with_tools=True))
            ] + state["messages"]
        response = answer_model.invoke(messages)

        return {"messages": response}

    # Define model and tool nodes
    graph.add_node("query", query)
    graph.add_node("retrieve_emails", ToolNode([retrieve_emails]))
    graph.add_node("answer", answer)
    graph.add_node("answer_with_citations", ToolNode([answer_with_citations]))

    # Route the user's input to the query model
    graph.add_edge(START, "query")

    # Add conditional edges from model to tools
    graph.add_conditional_edges(
        "query",
        tools_condition,
        {END: END, "tools": "retrieve_emails"},
    )
    graph.add_conditional_edges(
        "answer",
        tools_condition,
        {END: END, "tools": "answer_with_citations"},
    )

    # Add edge from the retrieval tool to the generating model
    graph.add_edge("retrieve_emails", "answer")

    # Done!
    return graph
