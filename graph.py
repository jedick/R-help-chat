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
from prompts import retrieve_prompt, answer_prompt, smollm3_tools_template
from mods.tool_calling_llm import ToolCallingLLM

# Local modules
from retriever import BuildRetriever


def ToolifySmolLM3(chat_model, system_message, system_message_suffix="", think=False):
    """
    Get a SmolLM3 model ready for bind_tools().
    """

    # Add /no_think flag to turn off thinking mode
    if not think:
        system_message = "/no_think\n" + system_message

    # NOTE: The first two nonblank lines are taken from the chat template for HuggingFaceTB/SmolLM3-3B
    # The rest are taken from the default system template for ToolCallingLLM
    tool_system_prompt_template = system_message + smollm3_tools_template

    class HuggingFaceWithTools(ToolCallingLLM, ChatHuggingFace):

        class Config:
            # Allows adding attributes dynamically
            extra = "allow"

    chat_model = HuggingFaceWithTools(
        llm=chat_model.llm,
        tool_system_prompt_template=tool_system_prompt_template,
        system_message_suffix=system_message_suffix,
    )

    # The "model" attribute is needed for ToolCallingLLM to print the response if it can't be parsed
    chat_model.model = chat_model.model_id + "_for_tools"

    return chat_model


def BuildGraph(
    chat_model,
    compute_location,
    search_type,
    top_k=6,
    think_retrieve=False,
    think_generate=False,
):
    """
    Build conversational RAG graph for email retrieval and answering with citations.

    Args:
        chat_model: LangChain chat model from GetChatModel()
        compute_location: cloud or edge (for retriever)
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
            compute_location, search_type, top_k, start_year, end_year
        )
        # For now, just add the months to the search query
        if months:
            search_query = search_query + " " + months
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
        query_model = ToolifySmolLM3(
            chat_model, retrieve_prompt(compute_location), "", think_retrieve
        ).bind_tools([retrieve_emails])
        generate_model = ToolifySmolLM3(
            chat_model, answer_prompt(), "", think_generate
        ).bind_tools([answer_with_citations])
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
            # Convert ToolMessage (from previous turns) to AIMessage
            # (avoids ValueError: Unknown message type: <class 'langchain_core.messages.tool.ToolMessage'>)
            messages = [
                AIMessage(msg.content) if type(msg) is ToolMessage else msg
                for msg in messages
            ]
        else:
            messages = [SystemMessage(retrieve_prompt(compute_location))] + state[
                "messages"
            ]
        response = query_model.invoke(messages)

        return {"messages": response}

    def generate(state: MessagesState):
        """Generates an answer with the chat model"""
        if is_edge:
            messages = state["messages"]
            # Copy the most recent HumanMessage to the end
            # (avoids ValueError: Last message must be a HumanMessage!)
            for msg in reversed(messages):
                if type(msg) is HumanMessage:
                    messages.append(msg)
            # Convert ToolMessage to AIMessage
            messages = [
                AIMessage(msg.content) if type(msg) is ToolMessage else msg
                for msg in messages
            ]
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
