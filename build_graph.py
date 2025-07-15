from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import List, Annotated, TypedDict
from langchain_huggingface import ChatHuggingFace
from tool_calling_llm import ToolCallingLLM
from dotenv import load_dotenv
import os


def ToolifySmolLM3(chat_model, system_message_start, think=False):
    """
    Get a SmolLM3 model ready for bind_tools().
    """

    # Add \no_think flag to turn off thinking mode
    if not think:
        system_message_start = "/no_think\n" + system_message_start

    # NOTE: The first two nonblank lines are taken from the chat template for HuggingFaceTB/SmolLM3-3B
    # The rest are taken from the default system template for ToolCallingLLM
    tool_system_prompt_template = (
        system_message_start
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
        think_retrieve: Whether to use thinking mode for retrieval (tool-calling)
        think_generate: Whether to use thinking mode for generation

    Example: RunGraph("What R functions are discussed?")

    Based on:
        https://python.langchain.com/docs/tutorials/qa_chat_history
        https://python.langchain.com/docs/how_to/qa_sources
    """

    # For tracing
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_PROJECT"] = "R-help-chat"
    # For LANGCHAIN_API_KEY
    load_dotenv(dotenv_path=".env", override=True)

    # Define start of system message, used in both respond_or_retrieve and generate
    system_message_start = (
        "You are a helpful RAG chatbot designed to answer questions about R programming. "
        "Do not ask the user for more information, but retrieve emails from the R-help mailing list archives. "
        "Summarize the retrieved emails to give an answer. "
        "To answer 'X or Y?', retrieve emails about X and Y to support your answer. "
        "Tell the user if you are unable to answer the question based on the information in the emails. "
        "It is more helpful to say that there is not enough information than to respond with your own ideas or suggestions. "
        "Do not give an answer based on your own knowledge or memory. "
        "For example, a question about macros should not be answered with 'knitr' and 'markdown' if those packages aren't described in the retrieved emails. "
        "Respond with 200 words maximum and 20 lines of code maximum. "
        "Cite the sender's name and date taken from the email headers. "
    )

    # Define retrieval tool
    # We propagate the retrieved documents as artifacts on the tool messages.
    # That makes it easy to pluck out the retrieved documents.
    # Below, we add them as an additional key in the state, for convenience.
    # Define the response format of the tool as "content_and_artifact":
    @tool(response_format="content_and_artifact")
    def retrieve_emails(query: str):
        """Retrieve emails related to a query from the R-help mailing list archives"""
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
        # Add an answer key with the final answer
        answer: str
        # Add a context key to the state to store retrieved documents
        context: List[Document]
        # Add a senders key that contains the cited senders
        senders: List[str]

    # Define response or retrieval step (entry point)
    # NOTE: This has to be ChatMessagesState, not MessagesState, to access step["context"]
    def respond_or_retrieve(state: ChatMessagesState):
        """Generate AI response or tool call for retrieval"""
        # Local models (ChatHuggingFace) have a "model_id" attribute
        if hasattr(chat_model, "model_id"):
            chat_model_for_tools = ToolifySmolLM3(
                chat_model, system_message_start, think_retrieve
            )
            chat_model_with_tools = chat_model_for_tools.bind_tools([retrieve_emails])
            response = chat_model_with_tools.invoke(state["messages"])
        else:
            chat_model_with_tools = chat_model.bind_tools([retrieve_emails])
            response = chat_model_with_tools.invoke(
                [SystemMessage(system_message_start)] + state["messages"]
            )
        # MessagesState appends messages to state instead of overwriting
        return {"messages": [response]}

    # Define retrieval step
    tools = ToolNode([retrieve_emails])

    # Desired schema for response
    class AnswerWithSources(TypedDict):
        """An answer to the question, with senders."""

        answer: str
        senders: Annotated[
            List[str],
            ...,
            "Senders of email messages used to answer the question, e.g. Duncan Murdoch, 2025-05-31",
        ]

    # Define generation step
    def generate(state: MessagesState):
        """Generate a response using the retrieved emails"""
        # Get generated ToolMessages
        recent_tool_messages = []
        for message in reversed(state["messages"]):
            if message.type == "tool":
                recent_tool_messages.append(message)
            else:
                break
        tool_messages = recent_tool_messages[::-1]
        # Pluck out the retrieved emails to be added to the state
        context = []
        for tool_message in tool_messages:
            context.extend(tool_message.artifact)

        # Generate system message
        docs_content = "\n\n".join(doc.content for doc in tool_messages)
        system_message_middle = "\n\n### Retrieved Emails:\n\n"
        # This is disabled until we figure out a better way to get a structured response
        if hasattr(chat_model, "model_id") and False:
            system_message_middle = """
            ### Additional Instructions

            You must respond with only a JSON object matching the following schema:

            {
              "answer": <An answer to the question>,
              "senders": <Senders of email messages used to answer the question, e.g. Duncan Murdoch, 2025-05-31>
            }

            ### Retrieved Emails:

            """
        system_message = system_message_start + system_message_middle + docs_content
        if hasattr(chat_model, "model_id") and not think_generate:
            # /no_think is needed to avoid parsing error (<think> .. </think> is not valid JSON)
            # TODO: Strip <think> output before parsing in langchain_huggingface/chat_models/huggingface.py
            system_message = "/no_think\n" + system_message
        # Combine conversation messages
        conversation_messages = [
            message
            for message in state["messages"]
            if message.type == "human"
            or (message.type == "ai" and not message.tool_calls)
        ]
        # Format system and conversation messages into prompt
        prompt = [SystemMessage(system_message)] + conversation_messages

        ## Run the "naked" chat model (keep this here for testing)
        # messages = chat_model.invoke(prompt)
        # return {"messages": messages, "context": context}

        # Setup the chat model for structured output
        if hasattr(chat_model, "model_id"):
            ## Our system prompt has instructions for formatting the output into the desired schema, so we use json_mode
            # structured_chat_model = chat_model.with_structured_output(
            #    AnswerWithSources, method="json_mode"
            # )
            # response = structured_chat_model.invoke(prompt)
            messages = chat_model.invoke(prompt)
            return {"messages": messages, "context": context}
        else:
            structured_chat_model = chat_model.with_structured_output(AnswerWithSources)
            response = structured_chat_model.invoke(prompt)
            # Add the answer to the state as an AIMessage
            answer = response["answer"]
            return {
                "messages": AIMessage(answer),
                "answer": answer,
                "context": context,
                "senders": response["senders"],
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
