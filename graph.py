from langchain_core.messages import SystemMessage, ToolMessage, HumanMessage, AIMessage
from langgraph.graph import START, END, MessagesState, StateGraph
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_huggingface import ChatHuggingFace
from typing import Optional
from dotenv import load_dotenv
import datetime
import os

# Local modules
from retriever import BuildRetriever
from prompts import retrieve_prompt, answer_prompt, gemma_tools_template
from mods.tool_calling_llm import ToolCallingLLM

# Local modules
from retriever import BuildRetriever

## For LANGCHAIN_API_KEY
# load_dotenv(dotenv_path=".env", override=True)
# os.environ["LANGSMITH_TRACING"] = "true"
# os.environ["LANGSMITH_PROJECT"] = "R-help-chat"


def print_messages_summary(messages, header):
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
    # (avoids SmolLM3 ValueError: Last message must be a HumanMessage!)
    if not type(messages[-1]) is HumanMessage:
        for msg in reversed(messages):
            if type(msg) is HumanMessage:
                messages.append(msg)
    # Convert tool output (ToolMessage) to AIMessage
    # (avoids SmolLM3 ValueError: Unknown message type: <class 'langchain_core.messages.tool.ToolMessage'>)
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


def ToolifyHF(chat_model, system_message, system_message_suffix="", think=False):
    """
    Get a Hugging Face model ready for bind_tools().
    """

    ## Add /no_think flag to turn off thinking mode (SmolLM3)
    # if not think:
    #    system_message = "/no_think\n" + system_message

    # Combine system prompt and tools template
    tool_system_prompt_template = system_message + gemma_tools_template

    class HuggingFaceWithTools(ToolCallingLLM, ChatHuggingFace):

        class Config:
            # Allows adding attributes dynamically
            extra = "allow"

    chat_model = HuggingFaceWithTools(
        llm=chat_model.llm,
        tool_system_prompt_template=tool_system_prompt_template,
        # Suffix is for any additional context (not templated)
        system_message_suffix=system_message_suffix,
    )

    # The "model" attribute is needed for ToolCallingLLM to print the response if it can't be parsed
    chat_model.model = chat_model.model_id + "_for_tools"

    return chat_model


def BuildGraph(
    chat_model,
    compute_mode,
    search_type,
    top_k=6,
    think_retrieve=False,
    think_generate=False,
):
    """
    Build conversational RAG graph for email retrieval and answering with citations.

    Args:
        chat_model: LangChain chat model from GetChatModel()
        compute_mode: cloud or edge (for retriever)
        search_type: dense, sparse, or hybrid (for retriever)
        top_k: number of documents to retrieve
        think_retrieve: Whether to use thinking mode for retrieval
        think_generate: Whether to use thinking mode for generation

    Based on:
        https://python.langchain.com/docs/how_to/qa_sources
        https://python.langchain.com/docs/tutorials/qa_chat_history
        https://python.langchain.com/docs/how_to/chatbots_memory/

    Usage Example:

        # Build graph with chat model
        from langchain_openai import ChatOpenAI
        chat_model = ChatOpenAI(model="gpt-4o-mini")
        graph = BuildGraph(chat_model, "cloud", "hybrid")

        # Add simple in-memory checkpointer
        from langgraph.checkpoint.memory import MemorySaver
        memory = MemorySaver()
        # Compile app and draw graph
        app = graph.compile(checkpointer=memory)
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
            compute_mode, search_type, top_k, start_year, end_year
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

    # Add tools to the edge or cloud chat model
    is_edge = hasattr(chat_model, "model_id")
    if is_edge:
        # For edge model (ChatHuggingFace)
        query_model = ToolifyHF(
            chat_model, retrieve_prompt(compute_mode), "", think_retrieve
        ).bind_tools([retrieve_emails])
        # Don't use answer_with_citations tool here because responses with Gemma are sometimes unparseable
        generate_model = ToolifyHF(
            chat_model, answer_prompt(with_tools=False), "", think_generate
        )
    else:
        # For cloud model (OpenAI API)
        query_model = chat_model.bind_tools([retrieve_emails])
        generate_model = chat_model.bind_tools([answer_with_citations])

    # Initialize the graph object
    graph = StateGraph(MessagesState)

    def query(state: MessagesState):
        """Queries the retriever with the chat model"""
        if is_edge:
            # Don't include the system message here because it's defined in ToolCallingLLM
            messages = state["messages"]
            # print_messages_summary(messages, "--- query: before normalization ---")
            messages = normalize_messages(messages)
            # print_messages_summary(messages, "--- query: after normalization ---")
        else:
            messages = [SystemMessage(retrieve_prompt(compute_mode))] + state[
                "messages"
            ]
        response = query_model.invoke(messages)

        return {"messages": response}

    def generate(state: MessagesState):
        """Generates an answer with the chat model"""
        if is_edge:
            messages = state["messages"]
            # print_messages_summary(messages, "--- generate: before normalization ---")
            messages = normalize_messages(messages)
            # print_messages_summary(messages, "--- generate: after normalization ---")
        else:
            messages = [SystemMessage(answer_prompt())] + state["messages"]
        response = generate_model.invoke(messages)

        return {"messages": response}

    # Define model and tool nodes
    graph.add_node("query", query)
    graph.add_node("generate", generate)
    graph.add_node("retrieve_emails", ToolNode([retrieve_emails]))
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
        "generate",
        tools_condition,
        {END: END, "tools": "answer_with_citations"},
    )

    # Add edge from the retrieval tool to the generating model
    graph.add_edge("retrieve_emails", "generate")

    # Done!
    return graph
