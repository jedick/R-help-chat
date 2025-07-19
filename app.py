import gradio as gr
from main import GetGraphAndConfig
from util import get_collection, get_sources, get_start_end_months
import asyncio
import torch
import uuid
import ast
import os

# Global state for chatbot app
graph = None
config = None
thread_id = None

# Initial settings, used to emit messages when settings are changed
COMPUTE = "cloud"
SEARCH = "hybrid"


def set_graph_config(compute_location, search_type, my_thread_id=None):
    """Helper to set the graph and config for the agent"""

    if my_thread_id is None:
        # This starts a new chat with a random thread ID
        global thread_id
        thread_id = uuid.uuid4()
        my_thread_id = thread_id
        # print(f"Generated thread_id: {thread_id}")

    global graph, config
    graph, config = GetGraphAndConfig(
        compute_location, search_type, thread_id=my_thread_id
    )

    global COMPUTE
    if not compute_location == COMPUTE:
        gr.Success(f"{compute_location}", duration=4, title=f"Set compute location!")
        COMPUTE = compute_location

    global SEARCH
    if not search_type == SEARCH:
        if search_type in ["dense", "sparse"]:
            message = f"{search_type}: up to 6 emails"
        elif search_type == "hybrid":
            message = "hybrid (dense + sparse): up to 3+3 emails"
        gr.Success(message, duration=4, title=f"Set search type!")
        SEARCH = search_type

    # Return the current value of thread_id
    return my_thread_id


async def interact_with_langchain_agent(input, messages, compute_location, search_type):

    # Set initial graph/config
    # print(f"Using thread_id: {thread_id}")
    if graph == None:
        set_graph_config(
            compute_location=COMPUTE, search_type=SEARCH, my_thread_id=thread_id
        )

    # This shows the user input as a chatbot message
    messages.append(gr.ChatMessage(role="user", content=input))
    # Return the messages for chatbot and chunks for emails and citations texboxes (blank at first)
    yield messages, [], []

    # Asynchronously stream graph steps for a single input
    # https://langchain-ai.lang.chat/langgraph/reference/graphs/#langgraph.graph.state.CompiledStateGraph
    async for step in graph.astream(
        # Appends the user input to the graph state
        {"messages": [{"role": "user", "content": input}]},
        config=config,
    ):

        # Get the node name and output chunk
        node, chunk = next(iter(step.items()))

        if node == "query":
            # Get the message (AIMessage class in LangChain)
            chunk_messages = chunk["messages"]
            # Look for a tool call
            if chunk_messages.tool_calls:
                tool_call = chunk_messages.tool_calls[0]
                # Show the tool call with arguments used
                args = tool_call["args"]
                content = args["query"]
                start_year = args["start_year"] if "start_year" in args else None
                end_year = args["end_year"] if "end_year" in args else None
                if start_year or end_year:
                    content = f"{content} ({start_year or ''} - {end_year or ''})"
                messages.append(
                    gr.ChatMessage(
                        role="assistant",
                        content=content,
                        metadata={"title": f"🔍 Running tool {tool_call['name']}"},
                    )
                )
            if chunk_messages.content:
                messages.append(
                    gr.ChatMessage(role="assistant", content=chunk_messages.content)
                )
            yield messages, [], []

        if node == "retrieve_emails":
            # Get the retrieved emails
            chunk_messages = chunk["messages"][0]
            retrieved_emails = chunk_messages.content.replace(
                "### Retrieved Emails:\n\n\n\n", ""
            ).split("--- --- --- --- Next Email --- --- --- ---\n\n")
            # print("--- retrieved_emails ---")
            # print(retrieved_emails)

            # Get the number of retrieved emails
            n_emails = len(retrieved_emails)
            # Get the list of source files (e.g. R-help/2024-December.txt) for retrieved emails
            month_list = [text.splitlines()[0] for text in retrieved_emails]
            # Format months (e.g. 2024-December) into text
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
            yield messages, retrieved_emails, []

        if node == "generate":
            chunk_messages = chunk["messages"]
            # Chat response without citations
            if chunk_messages.content:
                messages.append(
                    gr.ChatMessage(role="assistant", content=chunk_messages.content)
                )
            yield messages, None, []

        if node == "answer_with_citations":
            chunk_messages = chunk["messages"][0]
            # Parse the message for the answer and citations
            try:
                answer, citations = ast.literal_eval(chunk_messages.content)
            except:
                # In case we got an answer without citations
                answer = chunk_messages.content
                citations = None

            messages.append(gr.ChatMessage(role="assistant", content=answer))
            yield messages, None, citations


# Custom CSS for bottom alignment
css = """
.row-container {
    display: flex;
    align-items: flex-end; /* Align components at the bottom */
    gap: 10px; /* Optional: Add spacing between components */
}
"""

with gr.Blocks(
    title="R-help-chat",
    theme=gr.themes.Soft(font=["ui-sans-serif", "system-ui", "sans-serif"]),
    css=css,
) as demo:

    # Define components before rendering them
    compute_location = gr.Radio(
        choices=[
            "cloud",
            "edge" if torch.cuda.is_available() else "edge (not available)",
        ],
        value="cloud",
        label="Compute Location",
        info=(
            "edge models load in ca. 20 seconds; pop-up notifies when ready"
            if torch.cuda.is_available()
            else "edge models require GPU"
        ),
        interactive=torch.cuda.is_available(),
        render=False,
    )
    search_type = gr.Radio(
        choices=["dense", "sparse", "hybrid"],
        value="hybrid",
        label="Search Type",
        render=False,
    )
    input = gr.Textbox(
        lines=1,
        label="Your Question",
        info="Press Enter to submit",
        render=False,
    )
    show_examples = gr.Checkbox(
        value=False,
        label="💡 Example Questions",
        render=False,
    )
    chatbot = gr.Chatbot(
        type="messages",
        show_label=False,
        avatar_images=(
            None,
            "cloud.png",
        ),
        render=False,
    )

    # Make the interface
    with gr.Row(elem_classes=["row-container"]):
        with gr.Column(scale=2):
            gr.Markdown(
                """
            # 🤖 R-help-chat
            
            **Chat with the R-help mailing list archives.** Get AI-powered answers about R programming backed by email retrieval.<br>
            You can specify years or year ranges, ask follow-up questions, and get source citations.<br>
            **Privacy Note:** User questions and AI responses are logged using LangSmith for performance monitoring and detecting potential abuses.<br>
            Additionally, data sharing ("Input to the Services for Development Purposes") is enabled for the OpenAI API key used in this deployment.<br>
            If you are concerned about privacy, then do not use this app, or duplicate it to use your own API key or edge models on resources you control.
            """
            )
        with gr.Column(scale=1):
            # Add information about the system
            with gr.Accordion("ℹ️ About This System", open=False):

                # Get start and end months from database
                start, end = get_start_end_months(get_sources(compute_location.value))
                # Get number of emails (unique doc ids) in vector database
                collection = get_collection(compute_location.value)
                n_emails = len(set([m["doc_id"] for m in collection["metadatas"]]))
                gr.Markdown(
                    f"""
                    **R-help-chat** is a chat interface to the [R-help mailing list archives](https://stat.ethz.ch/pipermail/r-help/).
                    Current coverage is {n_emails} emails from {start} to {end}.
                    For technical details, see the [R-help-chat GitHub repo](https://github.com/jedick/R-help-chat).
                    
                    **Features:**
                    - **Date awareness**: The chat model knows today's date,
                    - **Tool usage**: queries the retrieval tool based on your question,
                    - **Chat generation**: answers based on retrieved emails, and
                    - **Source citations**: provides citations to emails.
                    
                    **Compute Location:**
                    - **cloud**: OpenAI API for embeddings and chat
                    - **edge**: [Nomic](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) embeddings and [SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) chat model
                    
                    **Search Types:**
                    - **dense**: Vector embeddings (semantic similarity)
                    - **sparse**: Keyword search with [BM25S](https://github.com/xhluca/bm25s) (good for function names)
                    - **hybrid**: Combination of dense and sparse
                    """
                )

    with gr.Row():
        with gr.Column(scale=2):
            input.render()
        with gr.Column(scale=1):
            with gr.Accordion("⚙️ Settings", open=False):
                compute_location.render()
                search_type.render()
            show_examples.render()

    with gr.Row():

        with gr.Column(scale=2):
            chatbot.render()

        with gr.Column(scale=1, visible=False) as examples:
            # Add some helpful examples
            example_questions = [
                "How to use lapply()?",
                "Recommend ML packages with examples",
                "When was has.HLC mentioned?",
                "Who discussed profiling in 2023?",
                "Any messages about installation problems in 2023-2024?",
                "Summarize last month's emails",
                "What is today's date?",
            ]
            gr.Examples(
                examples=[[q] for q in example_questions],
                inputs=[input],
                label="Click an example to fill the question box",
                elem_id="example-questions",
            )
            multi_turn_questions = [
                "Lookup emails that reference bugs.r-project.org in 2025",
                "Did those authors report bugs in 2024?",
            ]
            gr.Examples(
                examples=[[q] for q in multi_turn_questions],
                inputs=[input],
                label="Multi-turn example for asking follow-up questions",
                elem_id="example-questions",
            )

    with gr.Row():
        with gr.Column(scale=2):
            emails_textbox = gr.Textbox(
                label="Retrieved Emails", lines=2, visible=False
            )
        with gr.Column():
            citations_textbox = gr.Textbox(label="Citations", lines=2, visible=False)

    def visible(show):
        # Return updated visibility state for a component
        return gr.update(visible=show)

    # Show more info
    show_examples.change(visible, show_examples, examples)

    # Define state to keep track of global thread_id
    thread_id_state = gr.State(thread_id)
    # Define states for the output textboxes
    retrieved_emails = gr.State([])
    citations_text = gr.State([])

    def set_avatar(compute_location):
        if compute_location == "cloud":
            image_file = "images/cloud.png"
        if compute_location == "edge":
            image_file = "images/chip.png"
        return gr.update(
            avatar_images=(
                None,
                image_file,
            ),
        )

    # Set graph/config when compute location changes
    compute_location.change(
        set_graph_config,
        # This resets global thread_id and assigns it to thread_id_state
        [compute_location, search_type],
        [thread_id_state],
    ).then(
        # This changes the avatar for cloud or edge
        # TODO: make the change apply to only future messages
        set_avatar,
        compute_location,
        chatbot,
    )

    # Set graph/config when search type changes
    search_type.change(
        set_graph_config,
        # This reuses global thread_id
        [compute_location, search_type, thread_id_state],
        [thread_id_state],
    )

    # Submit input to the chatbot
    input.submit(
        interact_with_langchain_agent,
        [input, chatbot, compute_location, search_type],
        [chatbot, retrieved_emails, citations_text],
    )

    def update_emails(retrieved_emails, emails_textbox):
        if retrieved_emails is None:
            # This keeps the content of the textbox when the answer step occurs
            return emails_textbox, visible(True)
        elif not retrieved_emails == []:
            # This formats a non-empty list of retrieved emails for the textbox
            emails_text = "--- --- --- --- Next Email --- --- --- ---\n\n".join(
                retrieved_emails
            )
            return emails_text, visible(True)
        else:
            # This blanks out the textbox when a new chat is started
            return "", visible(False)

    # Update the emails textbox when ready
    retrieved_emails.change(
        update_emails,
        [retrieved_emails, emails_textbox],
        [emails_textbox, emails_textbox],
    )

    def update_citations(citations):
        if citations == []:
            # Blank out and hide the citations textbox when new input is submitted
            return "", visible(False)
        else:
            return citations, visible(True)

    # Update the citations textbox when ready
    citations_text.change(
        update_citations, citations_text, [citations_textbox, citations_textbox]
    )

if __name__ == "__main__":

    # Set allowed_paths to serve chatbot avatar images
    current_directory = os.getcwd()
    allowed_paths = [current_directory + "/images"]
    # Launch the Gradio app
    demo.launch(allowed_paths=allowed_paths)
