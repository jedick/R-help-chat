from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import List, Annotated, TypedDict
from langchain_huggingface import ChatHuggingFace
from tool_calling_llm import ToolCallingLLM


def ToolifySmolLM3(chat_model, system_message_prefix, think=False):
    """
    Get a SmolLM3 model ready for bind_tools().
    """

    # Add \no_think flag to turn off thinking mode
    if not think:
        system_message_prefix = "/no_think\n" + system_message_prefix

    # NOTE: The first two lines (after the prefix) are extracted from
    # tokenizer.apply_chat_template(xml_tools=) for HuggingFaceTB/SmolLM3-3B to activate tool calling
    # The remainder is taken from ToolCallingLLM
    tool_system_prompt_template = (
        system_message_prefix
        + """

    ### Tools

    You may call one or more functions to assist with the user query.

    You have access to the following tools:

    {tools}

    You must always select one of the above tools and respond with only a JSON object matching the following schema:

    {{
      "tool": <name of the selected tool>,
      "tool_input": <parameters for the selected tool, matching the tool's JSON schema>
    }}
    """
    )

    class HuggingFaceWithTools(ToolCallingLLM, ChatHuggingFace):

        class Config:
            # Allows adding attributes dynamically
            extra = "allow"

    chat_model = HuggingFaceWithTools(
        llm=chat_model.llm,
        tool_system_prompt_template=tool_system_prompt_template,
    )

    # The 'model' attribute is needed for ToolCallingLLM to print the response if it can't be parsed
    chat_model.model = chat_model.model_id + "_for_tools"

    return chat_model


def BuildGraph(retriever, chat_model, think_retrieve=False, think_generate=False):
    """
    Build graph for chat (conversational RAG with memory)

    Args:
        retriever: retriever instance from BuildRetriever()
        chat_model: LangChain chat model from GetChatModel()

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
        # Add a sources key that contains the cited sources
        sources: List[str]

    # Define response or retrieval step (entry point)
    # NOTE: This has to be ChatMessagesState, not MessagesState, to access step["context"]
    def respond_or_retrieve(state: ChatMessagesState):
        """Generate AI response or tool call for retrieval"""
        if chat_model.model_id == "HuggingFaceTB/SmolLM3-3B":
            chat_model_for_tools = ToolifySmolLM3(
                chat_model, system_message_prefix, think_retrieve
            )
            chat_model_with_tools = chat_model_for_tools.bind_tools([retrieve])
            response = chat_model_with_tools.invoke(state["messages"])
        else:
            chat_model_with_tools = chat_model.bind_tools([retrieve])
            response = chat_model_with_tools.invoke(
                [SystemMessage(system_message_prefix)] + state["messages"]
            )
        # MessagesState appends messages to state instead of overwriting
        return {"messages": [response]}

    # Define retrieval step
    tools = ToolNode([retrieve])

    # Desired schema for response
    class ResponseWithSources(TypedDict):
        """An answer to the question, with sources."""

        answer: str
        sources: Annotated[
            List[str],
            ...,
            "List of sources (sender's name and date in From: email headers) used to answer the question",
        ]

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
        # Pluck out the retrieved documents to be added to the state
        context = []
        for tool_message in tool_messages:
            context.extend(tool_message.artifact)

        # Format into prompt
        docs_content = "\n\n".join(doc.content for doc in tool_messages)
        system_message_with_context = (
            f"{system_message_prefix}"
            "\n\n"
            "### Retrieved Information:"
            f"{docs_content}"
        )
        if chat_model.model_id == "HuggingFaceTB/SmolLM3-3B" and not think_generate:
            system_message_with_context = "/no_think\n" + system_message_with_context
        conversation_messages = [
            message
            for message in state["messages"]
            if message.type in ("human", "system")
            or (message.type == "ai" and not message.tool_calls)
        ]
        prompt = [SystemMessage(system_message_with_context)] + conversation_messages

        ## Run the "naked" chat model (keep this here for testing)
        if chat_model.model_id == "HuggingFaceTB/SmolLM3-3B":
            # Currently SmolLM3 isn't setup for structured output
            messages = chat_model.invoke(prompt)
            return {"messages": messages, "context": context}
        else:
            # Run the chat model with structured output
            structured_chat_model = chat_model.with_structured_output(
                ResponseWithSources
            )
            response = structured_chat_model.invoke(prompt)
            # Add the answer to the state as an AIMessage
            message = AIMessage(response["answer"])
            return {
                "messages": message,
                "context": context,
                "sources": response["sources"],
            }

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
    graph_builder.set_finish_point("generate")

    return graph_builder
