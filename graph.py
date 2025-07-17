from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import List, Annotated, TypedDict
from langchain_huggingface import ChatHuggingFace
from tcl import ToolCallingLLM
from dotenv import load_dotenv
import os
import warnings

# Local modules
from prompts import retrieve_message, generate_message, smollm3_tools_template

# For tracing
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "R-help-chat"
# For LANGCHAIN_API_KEY
load_dotenv(dotenv_path=".env", override=True)


def ToolifySmolLM3(chat_model, system_message, system_message_suffix="", think=False):
    """
    Get a SmolLM3 model ready for bind_tools().
    """

    # Add \no_think flag to turn off thinking mode
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

    # The 'model' attribute is needed for ToolCallingLLM to print the response if it can't be parsed
    chat_model.model = chat_model.model_id + "_for_tools"

    return chat_model


def BuildGraph(retriever, chat_model, think_retrieve=True, think_generate=False):
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

    # Represent the state of our RAG application using a sequence of messages to enable:
    #   - Tool-calling features of chat models (rewrite user queries)
    #   - A "back-and-forth" conversational user experience
    class ChatMessagesState(MessagesState):
        # Add a context key to the state to store retrieved documents
        context: List[Document]
        # Add a citations key that contains the source citations
        citations: List[str]

    # Define retrieval tool with response format as "content_and_artifact"
    # (artifact lets us show the retrieved documents in the web interface)
    @tool(response_format="content_and_artifact")
    def retrieve_emails(query: str):
        """Retrieve emails related to a query from the R-help mailing list archives"""
        retrieved_docs = retriever.invoke(query)
        serialized = "\n\n--- --- --- --- Next Email --- --- --- ---".join(
            doc.page_content for doc in retrieved_docs
        )
        return serialized, retrieved_docs

    # Define response or retrieval step (entry point)
    # NOTE: This has to be ChatMessagesState, not MessagesState, to access step["context"]
    def respond_or_retrieve(state: ChatMessagesState):
        """Generate AI response or tool call for retrieval"""
        # Add tools to the edge or cloud chat model
        # Edge models (ChatHuggingFace) have a "model_id" attribute
        if hasattr(chat_model, "model_id"):
            model_for_tools = ToolifySmolLM3(
                chat_model, retrieve_message, "", think_retrieve
            )
            tooled_model = model_for_tools.bind_tools([retrieve_emails])
            invoke_messages = state["messages"]
        else:
            tooled_model = chat_model.bind_tools([retrieve_emails])
            invoke_messages = [SystemMessage(retrieve_message)] + state["messages"]
        response = tooled_model.invoke(invoke_messages)
        # MessagesState appends messages to state instead of overwriting
        return {"messages": [response]}

    # Define retrieval step
    tools = ToolNode([retrieve_emails])

    # Define generation step
    def generate(state: MessagesState):
        """Generate a response using the retrieved emails"""

        # Get generated ToolMessages
        tool_messages = [
            message for message in state["messages"] if message.type == "tool"
        ]
        # Format retrieved emails to add to prompt
        retrieved_emails = "\n\n### Retrieved Emails:\n\n" + "\n\n".join(
            doc.content for doc in tool_messages
        )
        # Format retrieved emails to add to state
        context = []
        for tool_message in tool_messages:
            context.extend(tool_message.artifact)
        # Combine conversation messages
        conversation_messages = [
            message
            for message in state["messages"]
            if message.type == "human"
            or (message.type == "ai" and not message.tool_calls)
        ]

        ## Run the "naked" chat model (keep this here for testing)
        # messages = chat_model.invoke([SystemMessage(generate_message + retrieved_emails)] + conversation_messages)
        # return {"messages": [messages], "context": context}

        # Setup the chat model for structured output
        if hasattr(chat_model, "model_id"):

            # Local model: .with_structured_output() isn't supported, but we can use a tool
            @tool(response_format="content_and_artifact")
            def answer_with_citations(answer: str, citations: str):
                """
                An answer to the question, with citations of the emails used (senders and dates)

                Args:
                    answer: An answer to the question
                    citations: Citations of emails used to answer the question, e.g. Duncan Murdoch, 2025-05-31
                """
                return answer, citations

            model_for_tools = ToolifySmolLM3(
                chat_model,
                # Add instruction to use tool
                generate_message
                + "Use answer_with_citations to respond with an answer and citations. ",
                # Tried adding retrieved emails to system message, but the model forgot about tool calling
                # retrieved_emails,
                "",
                think_generate,
            )
            tooled_model = model_for_tools.bind_tools([answer_with_citations])
            ## Putting the query and retrieved emails in separate messages might lead to more hallucinations ...
            # messages = tooled_model.invoke(conversation_messages + [HumanMessage(retrieved_emails)])
            # ... so let's keep them together
            conversation_content = "\n\n".join(
                [message.content for message in conversation_messages]
            )
            messages = tooled_model.invoke(
                [HumanMessage(conversation_content + retrieved_emails)]
            )
            # Extract the tool calls
            tool_calls = messages.tool_calls
            if tool_calls:
                args = tool_calls[0]["args"]
                new_state = {
                    "messages": [AIMessage(args["answer"])],
                    "citations": [
                        (
                            args["citations"]
                            if "citations" in args
                            else "No citations generated by chat model."
                        )
                    ],
                }
            else:
                new_state = {
                    "messages": [messages],
                    # citations = "No citations generated by chat model.",
                }
        else:
            # OpenAI API: we can use .with_structured_output() method
            # Desired schema for response
            class AnswerWithCitations(TypedDict):
                """Answer the question with citations of the emails used (senders and dates)."""

                answer: str
                citations: Annotated[
                    List[str],
                    ...,
                    "Citations of emails used to answer the question, e.g. Duncan Murdoch, 2025-05-31",
                ]

            structured_chat_model = chat_model.with_structured_output(
                AnswerWithCitations
            )
            # Invoke model with system and conversation messages
            response = structured_chat_model.invoke(
                [SystemMessage(generate_message + retrieved_emails)]
                + conversation_messages
            )

            new_state = {"messages": [AIMessage(response["answer"])]}
            # Sometimes OpenAI API returns an answer without citations, so test that it's present
            if "citations" in response:
                new_state["citations"] = response["citations"]

        # Add retrieved emails (tool artifacts) to the state
        new_state["context"] = context

        return new_state

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
