from langchain_core.messages import SystemMessage, ToolMessage, HumanMessage, AIMessage
from langgraph.graph import START, END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from typing import Optional
import datetime
import os

# Local modules
from retriever import BuildRetriever
from prompts import query_prompt, answer_prompt

# For tracing (disabled)
# os.environ["LANGSMITH_TRACING"] = "true"
# os.environ["LANGSMITH_PROJECT"] = "R-help-chat"


def BuildGraph(
    chat_model,
    search_type,
    top_k=6,
):
    """
    Build conversational RAG graph for email retrieval and answering with citations.

    Args:
        chat_model: LangChain chat model
        search_type: dense, sparse, or hybrid (for retriever)
        top_k: number of documents to retrieve

    Based on:
        https://python.langchain.com/docs/how_to/qa_sources
        https://python.langchain.com/docs/tutorials/qa_chat_history
        https://python.langchain.com/docs/how_to/chatbots_memory/

    Usage Example:

        # Build graph with chat model
        from langchain_openai import ChatOpenAI
        chat_model = ChatOpenAI(model="gpt-4o-mini")
        graph = BuildGraph(chat_model, "hybrid")

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
            search_query (str): Search query
            start_year (int, optional): Starting year for emails
            end_year (int, optional): Ending year for emails
            months (str, optional): One or more months separated by spaces
        """
        retriever = BuildRetriever(
            search_type,
            top_k,
            start_year,
            end_year,
        )
        # For now, just add the months to the search query
        if months:
            search_query = " ".join([search_query, months])
        # If the search query is empty, use the years
        if not search_query:
            search_query = " ".join([search_query, start_year, end_year])
        retrieved_docs = retriever.invoke(search_query)
        serialized = "\n\n--- --- --- --- Next Email --- --- --- ---".join(
            # Add file name (e.g. R-help/2024-December.txt) from source key
            "\n\n" + doc.metadata["source"] + doc.page_content
            for doc in retrieved_docs
        )
        retrieved_emails = (
            "### Retrieved Emails:" + serialized
            if serialized
            else "### No emails were retrieved"
        )
        return retrieved_emails

    @tool(parse_docstring=True)
    def answer_with_citations(answer: str, citations: str) -> str:
        """
        An answer to the question, with citations of the emails used (senders and dates).

        Args:
            answer (str): An answer to the question
            citations (str): Citations of emails used to answer the question, e.g. Jane Doe, 2025-07-04; John Smith, 2020-01-01
        """
        return answer, citations

    # Add tools to the chat model
    query_model = chat_model.bind_tools([retrieve_emails])
    answer_model = chat_model.bind_tools([answer_with_citations])

    # Initialize the graph object
    graph = StateGraph(MessagesState)

    def query(state: MessagesState):
        """Queries the retriever with the chat model"""
        messages = [SystemMessage(query_prompt())] + state["messages"]
        response = query_model.invoke(messages)

        return {"messages": response}

    def answer(state: MessagesState):
        """Generates an answer with the chat model"""
        messages = [SystemMessage(answer_prompt())] + state["messages"]
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
