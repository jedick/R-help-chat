import gradio as gr
from main import GetGraphAndConfig
import asyncio
import torch
from util import list_sources, get_start_end_months


def get_graph_and_config(compute_location, search_type):
    """Helper to get the graph and config for the agent"""
    return GetGraphAndConfig(compute_location, search_type)


# Global state for graph/config (recreated on config change)
graph = None
config = None


def set_graph_config(compute_location, search_type):
    """Helper to set the graph and config in the app"""

    global graph, config
    graph, config = get_graph_and_config(compute_location, search_type)

    # global COMPUTE, SEARCH
    # if not search_type == SEARCH:
    #    SEARCH = search_type
    if search_type in ["dense", "sparse"]:
        message = f"{search_type}: up to 6 emails"
    elif search_type == "hybrid":
        message = "hybrid (dense + sparse): up to 3+3 emails"
    gr.Success(message, duration=4, title=f"Set search type!")


async def interact_with_langchain_agent(query, messages, compute_location, search_type):

    # Set initial graph/config
    if graph == None:
        set_graph_config(compute_location, search_type)

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
            # Look for a tool call
            if message.tool_calls:
                tool_call = message.tool_calls[0]
                messages.append(
                    gr.ChatMessage(
                        role="assistant",
                        content=tool_call["args"]["query"],
                        metadata={"title": f"🔍 Running tool {tool_call['name']}"},
                    )
                )
            else:
                # A response with no tool call
                messages.append(
                    gr.ChatMessage(role="assistant", content=message.content)
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


def clear_all():
    return [], "", "", ""


font = ["ui-sans-serif", "system-ui", "sans-serif"]

with gr.Blocks(title="R-help-chat", theme=gr.themes.Soft(font=font)) as demo:

    # Define components before rendering them
    compute_location = gr.Radio(
        choices=[
            "cloud",
            "edge" if torch.cuda.is_available() else "edge (not available)",
        ],
        value="cloud",
        label="Compute Location",
        interactive=torch.cuda.is_available(),
        render=False,
    )
    search_type = gr.Radio(
        choices=["dense", "sparse", "hybrid"],
        value="hybrid",
        label="Search Type",
        render=False,
    )
    query = gr.Textbox(
        lines=1,
        label="Your Question",
        info="Press Enter to submit",
        render=False,
    )
    help = gr.Checkbox(
        value=False,
        label="❓ Help",
        render=False,
    )

    # Make the interface
    gr.Markdown(
        """
    # 🤖 R-help-chat
    
    Chat with the R-help mailing list archives. Get AI-powered answers about R programming based on discussions from the R-help community.
    """
    )

    with gr.Row():
        with gr.Column(scale=2):
            query.render()
        with gr.Column(scale=1):
            with gr.Accordion("⚙️ Settings", open=False):
                compute_location.render()
                search_type.render()
            help.render()
            ## Add a clear button
            # gr.Button("Clear Chat").click(clear_all, None, [chatbot, query, citations, emails])

    with gr.Row():

        with gr.Column(scale=2):
            # The chatbot interface
            chatbot = gr.Chatbot(type="messages", show_label=False)

        with gr.Column(scale=1, visible=False) as about:
            # Add some helpful examples
            with gr.Accordion(
                "💡 Example Questions", open=True, visible=False
            ) as examples:
                example_questions = [
                    "How can I get a named argument from '...'?",
                    "How to print line numbers where errors occur?",
                    "Who discussed ways to handle missing values?",
                    "Were there bugs mentioned last month?",
                    "When was has.HLC mentioned?",
                ]
                gr.Examples(
                    examples=[[q] for q in example_questions],
                    inputs=[query],
                    label="Click an example to fill the question box",
                    elem_id="example-questions",
                )
            # Add information about the system
            with gr.Accordion("ℹ️ About This System", open=True):

                # Get start and end months from database
                start, end = get_start_end_months(list_sources(compute_location.value))
                gr.Markdown(
                    f"""
                    **R-help-chat** is a chat interface to the [R-help mailing list archives](https://stat.ethz.ch/pipermail/r-help/).
                    Current coverage is from {start} to {end}.
                    For technical details, see the [R-help-chat GitHub repo](https://github.com/jedick/R-help-chat).
                    
                    **Features:**
                    - **Tool usage**: LLM rewrites your query for search
                    - **Hybrid retrieval**: Combines dense and sparse search
                    - **Chat generation**: Answers based on retrieved emails
                    - **Source citations**: Provides citations to emails
                    
                    **Compute Location:**
                    - **cloud**: Uses OpenAI API (requires API key)
                    - **edge**: Uses edge models (requires GPU)
                    
                    **Search Types:**
                    - **dense**: Vector embeddings (semantic similarity)
                    - **sparse**: Keyword search (good for function names)
                    - **hybrid**: Combination of dense and sparse
                    """
                )

    with gr.Row():
        with gr.Column(scale=2):
            emails = gr.Textbox(label="Retrieved Emails", lines=2, visible=False)
        with gr.Column(visible=False) as citations_column:
            citations = gr.Textbox(label="Citations", lines=2)

    def visible(show):
        # Return updated visibility state for a component
        return gr.update(visible=show)

    # Show more info
    help.change(visible, help, about)
    help.change(visible, help, examples)

    # Define states for the retrieve and generate chunks
    retrieve_chunk = gr.State([])
    generate_chunk = gr.State([])

    # Set graph/config when search type changes
    search_type.change(
        set_graph_config,
        [compute_location, search_type],
        None,
    )

    # Submit a query to the chatbot
    query.submit(
        interact_with_langchain_agent,
        [query, chatbot, compute_location, search_type],
        [chatbot, retrieve_chunk, generate_chunk],
    )

    def update_emails(chunk, emails):
        if chunk is None:
            # This keeps the retrieved emails when the generate step is run
            return emails, visible(True)
        elif not chunk == []:
            # This gets the retrieved emails from a non-empty retrieve_chunk
            return chunk["messages"][0].content, visible(True)
        else:
            # This blanks out the textbox when a new chat is started
            return "", visible(False)

    # Update the emails when ready
    retrieve_chunk.change(update_emails, [retrieve_chunk, emails], [emails, emails])

    def update_citations(chunk):
        # No response yet
        if not chunk == []:
            # Response with no citations
            if not chunk["citations"] == []:
                # Format citations
                citations = "; ".join(chunk["citations"])
                return citations, visible(True)
        return "", visible(False)

    # Update the citations when ready, and blank it out when a new query is submitted
    generate_chunk.change(
        update_citations, generate_chunk, [citations, citations_column]
    )

if __name__ == "__main__":
    demo.launch()
