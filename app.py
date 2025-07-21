import gradio as gr
from main import GetChatModel
from graph import BuildGraph
from langgraph.checkpoint.memory import MemorySaver
from util import get_collection, get_sources, get_start_end_months
import spaces
import torch
import uuid
import ast
import os

# Keep LangChain graph in a global variable
# TODO: Use session state for non-deepcopyable objects
# https://www.gradio.app/guides/state-in-blocks#session-state

graph = None


def set_graph(compute_location, search_type):
    """Helper to set the graph for the workflow"""

    # Get the chat model and build the graph
    chat_model = GetChatModel(compute_location)
    graph_builder = BuildGraph(chat_model, compute_location, search_type)
    # Compile the graph with an in-memory checkpointer
    memory = MemorySaver()
    global graph
    graph = graph_builder.compile(checkpointer=memory)
    print(f"Set graph for {compute_location}, {search_type}!")


async def run_workflow(messages, input, thread_id):
    """The main function to run the chat workflow"""

    global graph
    if graph is None:
        set_graph(
            compute_location=compute_location.value, search_type=search_type.value
        )

    print(f"Using thread_id: {thread_id}")

    # Display the user input in the chatbot interface
    messages.append(gr.ChatMessage(role="user", content=input))
    # Return the messages for chatbot and chunks for emails and citations texboxes (blank at first)
    yield messages, [], []

    # Asynchronously stream graph steps for a single input
    # https://langchain-ai.lang.chat/langgraph/reference/graphs/#langgraph.graph.state.CompiledStateGraph
    async for step in graph.astream(
        # Appends the user input to the graph state
        {"messages": [{"role": "user", "content": input}]},
        config={"configurable": {"thread_id": thread_id}},
    ):

        # Get the node name and output chunk
        node, chunk = next(iter(step.items()))

        if node == "query":
            # Get the message (AIMessage class in LangChain)
            chunk_messages = chunk["messages"]
            # Look for tool calls
            if chunk_messages.tool_calls:
                # Loop over tool calls
                for tool_call in chunk_messages.tool_calls:
                    # Show the tool call with arguments used
                    args = tool_call["args"]
                    content = args["search_query"]
                    start_year = args["start_year"] if "start_year" in args else None
                    end_year = args["end_year"] if "end_year" in args else None
                    if start_year or end_year:
                        content = f"{content} ({start_year or ''} - {end_year or ''})"
                    if "months" in args:
                        content = f"{content} {args['months']}"
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
            chunk_messages = chunk["messages"]
            # Loop over tool calls
            count = 0
            retrieved_emails = []
            for message in chunk_messages:
                count += 1
                # Get the retrieved emails as a list
                email_list = message.content.replace(
                    "### Retrieved Emails:\n\n\n\n", ""
                ).split("--- --- --- --- Next Email --- --- --- ---\n\n")
                # Get the list of source files (e.g. R-help/2024-December.txt) for retrieved emails
                month_list = [text.splitlines()[0] for text in email_list]
                # Format months (e.g. 2024-December) into text
                month_text = (
                    ", ".join(month_list).replace("R-help/", "").replace(".txt", "")
                )
                # Get the number of retrieved emails
                n_emails = len(email_list)
                title = f"🛒 Retrieved {n_emails} emails"
                if email_list[0] == "### No emails were retrieved":
                    title = "❌ Retrieved 0 emails"
                messages.append(
                    gr.ChatMessage(
                        role="assistant",
                        content=month_text,
                        metadata={"title": title},
                    )
                )
                # Format the retrieved emails with Tool Call heading
                retrieved_emails.append(
                    message.content.replace(
                        "### Retrieved Emails:\n\n\n\n",
                        f"### ### ### ### Tool Call {count} ### ### ### ###\n\n",
                    )
                )
            # Combine all the Tool Call results
            retrieved_emails = "\n\n".join(retrieved_emails)
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


# Wrapper functions for the workflow, with @spaces.GPU decorator for edge


async def run_workflow_cloud(*args):
    async for value in run_workflow(*args):
        yield value


@spaces.GPU(duration=120)
async def run_workflow_edge(*args):
    async for value in run_workflow(*args):
        yield value


# Function to route the Gradio call to the correct workflow function


async def to_workflow(compute_location, *args):
    if compute_location == "cloud":
        async for value in run_workflow_cloud(*args):
            yield value
    if compute_location == "edge":
        async for value in run_workflow_edge(*args):
            yield value


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

    # -----------------
    # Define components
    # -----------------

    compute_location = gr.Radio(
        choices=[
            "cloud",
            "edge" if torch.cuda.is_available() else "edge (not available)",
        ],
        value="cloud",
        label="Compute Location",
        info=(
            "The edge model is experimental and may produce lower-quality answers. Pop-up will notify when it's ready (loading time is about 20 seconds)."
            if torch.cuda.is_available()
            else "NOTE: edge model requires GPU"
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
            "images/cloud.png",
        ),
        render=False,
    )

    # ------------------
    # Make the interface
    # ------------------

    def get_intro_text():
        # Get start and end months from database
        start, end = get_start_end_months(get_sources(compute_location.value))
        intro = f"""<!-- # 🤖 R-help-chat -->
            
            **Chat with the R-help mailing list archives.** Get AI-powered answers about R programming backed by email retrieval.<br>
            Use natural langauge to ask R-related questions including years or year ranges (coverage is {start} to {end}).<br>
            The chat model can rewrite your query for retrieval, make multiple retrievals in one turn, and provide source citations.<br>
            You can ask follow-up questions or clear the chat if you want to start over.<br>
            **_Answers may be incorrect._**<br>
            **Privacy Notice:** User questions and AI responses are logged for usage and performance monitoring.<br>
            Additionally, data sharing ("Input to the Services for Development Purposes") is enabled for the OpenAI API key used in this deployment.<br>
            """
        return intro

    with gr.Row(elem_classes=["row-container"]):
        with gr.Column(scale=2):
            intro = gr.Markdown(get_intro_text())
        with gr.Column(scale=1):
            # Add information about the system
            with gr.Accordion("ℹ️ About This System", open=False):

                # Get number of emails (unique doc ids) in vector database
                collection = get_collection(compute_location.value)
                n_emails = len(set([m["doc_id"] for m in collection["metadatas"]]))
                gr.Markdown(
                    f"""
                    **R-help-chat** is a chat interface to {n_emails} emails from the [R-help mailing list archives](https://stat.ethz.ch/pipermail/r-help/).
                    For technical details, see the [GitHub repository](https://github.com/jedick/R-help-chat).

                    **Open Source:** The source code for this app is distributed under the terms of the MIT license.
                    
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
                # "What is today's date?",
                "Summarize emails from the last two months",
                "What plotmath examples have been discussed?",
                "When was has.HLC mentioned?",
                "Who discussed profiling in 2023?",
                "Any messages about installation problems in 2023-2024?",
            ]
            gr.Examples(
                examples=[[q] for q in example_questions],
                inputs=[input],
                label="Click an example to fill the question box",
                elem_id="example-questions",
            )
            multi_tool_questions = [
                "Speed differences between lapply and for loops",
                "Compare usage of pipe operator between 2022 and 2024",
            ]
            gr.Examples(
                examples=[[q] for q in multi_tool_questions],
                inputs=[input],
                label="Example prompts for multiple retrievals",
                elem_id="example-questions",
            )
            multi_turn_questions = [
                "Lookup emails that reference bugs.r-project.org in 2025",
                "Did those authors report bugs before 2025?",
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
                label="Retrieved Emails",
                lines=10,
                visible=False,
                info="Hint: Look for 'Tool Call' and 'Next Email' separators. Quoted lines (starting with '>') are removed before indexing.",
            )
        with gr.Column():
            citations_textbox = gr.Textbox(label="Citations", lines=2, visible=False)

    # ------------
    # Set up state
    # ------------

    def generate_thread_id():
        """Generate a new thread ID"""
        thread_id = uuid.uuid4()
        print(f"Generated thread_id: {thread_id}")
        return thread_id

    # Define thread_id variable
    thread_id = gr.State(generate_thread_id())

    # Define states for the output textboxes
    retrieved_emails = gr.State([])
    citations_text = gr.State([])

    # -------------
    # Handle events
    # -------------

    # Start a new thread when the user presses the clear (trash) button
    # https://github.com/gradio-app/gradio/issues/9722
    chatbot.clear(generate_thread_id, outputs=[thread_id], api_name=False)

    def visible(show):
        """Return updated visibility state for a component"""
        return gr.update(visible=show)

    # Show examples
    show_examples.change(visible, show_examples, examples, api_name=False)

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

    def notify_compute(compute_location):
        """Notify when compute location changes"""
        gr.Success(f"{compute_location}", duration=4, title=f"Model is ready!")

    # Get graph when compute location changes
    compute_location.change(
        set_graph,
        [compute_location, search_type],
        api_name=False,
    ).then(
        notify_compute,
        [compute_location],
        api_name=False,
    ).then(
        # This sets the avatar for cloud or edge
        # TODO: make the change apply to only future messages
        set_avatar,
        [compute_location],
        [chatbot],
        api_name=False,
    )

    def notify_search(search_type):
        """Notify when search type changes"""
        if search_type in ["dense", "sparse"]:
            message = f"{search_type}: up to 6 emails"
        elif search_type == "hybrid":
            message = "hybrid (dense + sparse): up to 3+3 emails"
        gr.Success(message, duration=4, title=f"Set search type!")

    # Get graph when search type changes
    search_type.change(
        set_graph,
        [compute_location, search_type],
        api_name=False,
    ).then(
        notify_search,
        [search_type],
        api_name=False,
    )

    # Submit input to the chatbot
    input.submit(
        to_workflow,
        [compute_location, chatbot, input, thread_id],
        [chatbot, retrieved_emails, citations_text],
        api_name=False,
    )

    def update_emails(retrieved_emails, emails_textbox):
        if retrieved_emails is None:
            # This keeps the content of the textbox when the answer step occurs
            return emails_textbox, visible(True)
        elif not retrieved_emails == []:
            # This adds the retrieved emails to the textbox
            return retrieved_emails, visible(True)
        else:
            # This blanks out the textbox when a new chat is started
            return "", visible(False)

    # Update the emails textbox when ready
    retrieved_emails.change(
        update_emails,
        [retrieved_emails, emails_textbox],
        [emails_textbox, emails_textbox],
        api_name=False,
    )

    def update_citations(citations):
        if citations == []:
            # Blank out and hide the citations textbox when new input is submitted
            return "", visible(False)
        else:
            return citations, visible(True)

    # Update the citations textbox when ready
    citations_text.change(
        update_citations,
        citations_text,
        [citations_textbox, citations_textbox],
        api_name=False,
    )


if __name__ == "__main__":

    # Set allowed_paths to serve chatbot avatar images
    current_directory = os.getcwd()
    allowed_paths = [current_directory + "/images"]
    # Launch the Gradio app
    demo.launch(allowed_paths=allowed_paths)
