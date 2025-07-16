import gradio as gr
from main import GetGraphAndConfig
import asyncio


def get_graph_and_config(compute_location, search_type):
    """Helper to get the graph and config for the agent"""
    return GetGraphAndConfig(compute_location, search_type)


# Global state for graph/config (recreated on config change)
graph = None
config = None


def set_graph_config(compute_location, search_type, reranking):
    """Helper to set the graph and config in the app"""

    # Add _rr to search type for reranking
    if reranking and not search_type == "dense":
        search_type = search_type + "_rr"

    global graph, config
    graph, config = get_graph_and_config(compute_location, search_type)

    # global COMPUTE, SEARCH
    # if not search_type == SEARCH:
    #    SEARCH = search_type
    message = "hello"
    if search_type in ["dense", "sparse", "sparse_rr"]:
        message = f"{search_type}: up to 6 emails"
    elif search_type == "hybrid":
        message = "hybrid (dense + sparse): up to 3+3 emails"
    elif search_type == "hybrid_rr":
        message = "hybrid_rr (dense + sparse + sparse_rr): up to 2+2+2 emails"
    gr.Success(message, duration=4, title=f"Set search type!")


async def interact_with_langchain_agent(
    query, messages, compute_location, search_type, reranking
):

    # Set initial graph/config
    if graph == None:
        set_graph_config(compute_location, search_type, reranking)

    print("hello")
    print(graph)

    # This shows the user query as a chatbot message
    messages.append(gr.ChatMessage(role="user", content=query))
    # Return the messages for chatbot and chunks for emails and citations texboxes (blank at first)
    yield messages, [], []

    # Asynchronously stream graph steps for a single input
    # https://langchain-ai.lang.chat/langgraph/reference/graphs/#langgraph.graph.state.CompiledStateGraph
    async for step in graph.astream(
        # Appends the user query to the graph state
        {"messages": [{"role": "user", "content": query}]},
        config=config,
    ):
        # Get the node name and output chunk
        node, chunk = next(iter(step.items()))
        if node == "respond_or_retrieve":
            # Get the message (AIMessage class in LangChain)
            message = chunk["messages"][0]
            if hasattr(message, "tool_calls"):
                tool_call = message.tool_calls[0]
                messages.append(
                    gr.ChatMessage(
                        role="assistant",
                        content=tool_call["args"]["query"],
                        metadata={"title": f"🔍 Running tool {tool_call['name']}"},
                    )
                )
                yield messages, [], []
        if node == "tools":
            # Get the artifact of the retrieve_emails tool
            artifact = chunk["messages"][0].artifact
            # Get the number of retrieved emails
            n_emails = len(artifact)
            # Get the list of months for retrieved emails
            month_list = [email.metadata["source"] for email in artifact]
            # Format into text
            month_text = (
                ", ".join(month_list).replace("R-help/", "").replace(".txt", "")
            )
            messages.append(
                gr.ChatMessage(
                    role="assistant",
                    content=month_text,
                    metadata={"title": f"📤 Retrieved {n_emails} emails"},
                )
            )
            yield messages, chunk, []
        if node == "generate":
            messages.append(gr.ChatMessage(role="assistant", content=chunk["answer"]))
            yield messages, None, chunk


def update_emails(chunk, emails):
    if chunk is None:
        # This keeps the retrieved emails when the generate step is run
        return emails
    elif chunk != []:
        # This gets the retrieved emails from a non-empty retrieve_chunk
        return chunk["messages"][0].content
    else:
        # This blanks out the textbox when a new chat is started
        return ""


def update_citations(chunk):
    if chunk and "citations" in chunk:
        return chunk["citations"]
    return ""


def clear_all():
    return [], "", "", ""


with gr.Blocks(title="R-help-chat", theme=gr.themes.Soft()) as demo:

    # Make the interface
    gr.Markdown(
        """
    # 🤖 R-help-chat
    
    Chat with the R-help mailing list archives using AI. Ask questions about R programming and get answers based on real discussions from the R-help community.
    """
    )
    with gr.Row():
        compute_location = gr.Radio(
            choices=["cloud", "edge"],
            value="cloud",
            label="Compute Location",
            info="Cloud: OpenAI API, Edge: Local models (requires GPU)",
        )
        search_type = gr.Radio(
            choices=["dense", "sparse", "hybrid"],
            value="hybrid",
            label="Search Type",
            info="Different retrieval strategies",
        )
        reranking = gr.Checkbox(
            value=True,
            label="Reranking",
            info="Option for sparse search",
        )
    chatbot = gr.Chatbot(type="messages", label="Chatbot")
    with gr.Row():
        query = gr.Textbox(lines=1, label="Your Question")
    with gr.Row():
        citations = gr.Textbox(label="Citations", interactive=False)
    with gr.Row():
        emails = gr.Textbox(label="Retrieved emails", interactive=False)

    # Define states for the retrieve and generate chunks
    retrieve_chunk = gr.State([])
    generate_chunk = gr.State([])

    # Set graph/config when search type changes
    search_type.change(
        set_graph_config,
        [compute_location, search_type, reranking],
        None,
    )
    reranking.change(
        set_graph_config,
        [compute_location, search_type, reranking],
        None,
    )

    # Submit a query to the chatbot
    query.submit(
        interact_with_langchain_agent,
        [query, chatbot, compute_location, search_type, reranking],
        [chatbot, retrieve_chunk, generate_chunk],
    )
    # Update the emails when ready
    retrieve_chunk.change(update_emails, [retrieve_chunk, emails], emails)
    # Update the citations when ready, and blank it out when a new query is submitted
    generate_chunk.change(update_citations, generate_chunk, citations)
    # Add a clear button
    gr.Button("Clear Chat").click(clear_all, None, [chatbot, query, citations, emails])

if __name__ == "__main__":
    demo.launch()
