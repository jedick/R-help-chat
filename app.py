import gradio as gr
from main import GetChatModel
from graph import BuildGraph
from retriever import db_dir
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

# from util import get_collection, get_sources, get_start_end_months
from git import Repo
import zipfile
import spaces
import torch
import uuid
import ast
import os

# Global settings for compute_location and search_type
COMPUTE = "cloud"
search_type = "hybrid"

# Load LANGCHAIN_API_KEY (for local deployment)
load_dotenv(dotenv_path=".env", override=True)
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "R-help-chat"

# Check for GPU
if COMPUTE == "edge":
    if not torch.cuda.is_available():
        raise Exception("Can't use edge compute with no GPU")

# Keep LangChain graph in a global variable (shared across sessions)
graph_edge = None
graph_cloud = None


def run_workflow(chatbot, input, thread_id):
    """The main function to run the chat workflow"""

    # Get global graph for compute location
    global graph_edge, graph_cloud
    if COMPUTE == "edge":
        graph = graph_edge
    if COMPUTE == "cloud":
        graph = graph_cloud

    if graph is None:
        # Notify when we're loading the edge model because it takes some time
        if COMPUTE == "edge":
            gr.Warning(
                f"Please wait for the edge model to load",
                duration=15,
                title=f"Model loading...",
            )
        # Get the chat model and build the graph
        chat_model = GetChatModel(COMPUTE)
        graph_builder = BuildGraph(chat_model, COMPUTE, search_type)
        # Compile the graph with an in-memory checkpointer
        memory = MemorySaver()
        graph = graph_builder.compile(checkpointer=memory)
        # Set global graph for compute location
        if COMPUTE == "edge":
            graph_edge = graph
        if COMPUTE == "cloud":
            graph_cloud = graph

    # Notify when model finishes loading
    gr.Success(f"{COMPUTE}", duration=4, title=f"Model loaded!")
    print(f"Set graph for {COMPUTE}, {search_type}!")

    print(f"Using thread_id: {thread_id}")

    # Display the user input in the chatbot interface
    chatbot.append(gr.ChatMessage(role="user", content=input))
    # Return the chatbot messages and empty lists for emails and citations texboxes
    yield chatbot, [], []

    # Asynchronously stream graph steps for a single input
    # https://langchain-ai.lang.chat/langgraph/reference/graphs/#langgraph.graph.state.CompiledStateGraph
    for step in graph.stream(
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
                    content = args["search_query"] if "search_query" in args else ""
                    start_year = args["start_year"] if "start_year" in args else None
                    end_year = args["end_year"] if "end_year" in args else None
                    if start_year or end_year:
                        content = f"{content} ({start_year or ''} - {end_year or ''})"
                    if "months" in args:
                        content = f"{content} {args['months']}"
                    chatbot.append(
                        gr.ChatMessage(
                            role="assistant",
                            content=content,
                            metadata={"title": f"🔍 Running tool {tool_call['name']}"},
                        )
                    )
            if chunk_messages.content:
                chatbot.append(
                    gr.ChatMessage(role="assistant", content=chunk_messages.content)
                )
            yield chatbot, [], []

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
                chatbot.append(
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
            yield chatbot, retrieved_emails, []

        if node == "generate":
            chunk_messages = chunk["messages"]
            # Chat response without citations
            if chunk_messages.content:
                chatbot.append(
                    gr.ChatMessage(role="assistant", content=chunk_messages.content)
                )
            # None is used for no change to the retrieved emails textbox
            yield chatbot, None, []

        if node == "answer_with_citations":
            chunk_messages = chunk["messages"][0]
            # Parse the message for the answer and citations
            try:
                answer, citations = ast.literal_eval(chunk_messages.content)
            except:
                # In case we got an answer without citations
                answer = chunk_messages.content
                citations = None

            chatbot.append(gr.ChatMessage(role="assistant", content=answer))
            yield chatbot, None, citations


def to_workflow(*args):
    """Wrapper function to call function with or without @spaces.GPU"""
    if COMPUTE == "edge":
        for value in run_workflow_edge(*args):
            yield value
    if COMPUTE == "cloud":
        for value in run_workflow_cloud(*args):
            yield value


@spaces.GPU(duration=120)
def run_workflow_edge(*args):
    for value in run_workflow(*args):
        yield value


def run_workflow_cloud(*args):
    for value in run_workflow(*args):
        yield value


# Custom CSS for bottom alignment
css = """
.row-container {
    display: flex;
    align-items: flex-end; /* Align components at the bottom */
    gap: 10px; /* Add spacing between components */
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
        value=COMPUTE,
        label="Compute Location",
        info=(None if torch.cuda.is_available() else "NOTE: edge model requires GPU"),
        interactive=torch.cuda.is_available(),
        render=False,
    )

    input = gr.Textbox(
        lines=1,
        label="Your Question",
        info="Press Enter to submit",
        render=False,
    )
    downloading = gr.Textbox(
        lines=1,
        label="Downloading Data, Please Wait",
        visible=False,
        render=False,
    )
    extracting = gr.Textbox(
        lines=1,
        label="Extracting Data, Please Wait",
        visible=False,
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
            (
                "images/cloud.png"
                if compute_location.value == "cloud"
                else "images/chip.png"
            ),
        ),
        show_copy_all_button=True,
        render=False,
    )

    # ------------------
    # Make the interface
    # ------------------

    def get_intro_text():
        ## Get start and end months from database
        # start, end = get_start_end_months(get_sources(compute_location.value))
        intro = f"""<!-- # 🤖 R-help-chat -->
            ## 🇷🤝💬 R-help-chat
            
            **Chat with the [R-help mailing list archives]((https://stat.ethz.ch/pipermail/r-help/)).** Get AI-powered answers about R programming backed by email retrieval.
            An LLM turns your question into a search query, including year ranges.
            You can ask follow-up questions with the chat history as context.
            ➡️ To clear the chat history and start a new chat, press the 🗑️ trash button.<br>
            **_Answers may be incorrect._**<br>
            """
        return intro

    def get_info_text(compute_location):
        info_prefix = """
            **Features:** conversational RAG, today's date, email database (*start* to *end*), hybrid search (dense+sparse),
            query analysis, multiple tool calls (cloud model), answer with citations.
            **Tech:** LangChain + Hugging Face + Gradio; ChromaDB and BM25S-based retrievers.<br>
            """
        if compute_location.startswith("cloud"):
            info_text = f"""{info_prefix}
            📍 This is the **cloud** version, using the OpenAI API<br>
            ✨ gpt-4o-mini<br>
            ⚠️ **_Privacy Notice_**: Data sharing with OpenAI is enabled, and all interactions are logged<br>
            🏠 See the project's [GitHub repository](https://github.com/jedick/R-help-chat)
            """
        if compute_location.startswith("edge"):
            info_text = f"""{info_prefix}
            📍 This is the **edge** version, using [ZeroGPU](https://huggingface.co/docs/hub/spaces-zerogpu) hardware<br>
            ✨ Nomic embeddings and Gemma-3 LLM<br>
            ⚠️ **_Privacy Notice_**: All interactions are logged<br>
            🏠 See the project's [GitHub repository](https://github.com/jedick/R-help-chat)
            """
        return info_text

    with gr.Row(elem_classes=["row-container"]):
        with gr.Column(scale=2):
            with gr.Row(elem_classes=["row-container"]):
                with gr.Column(scale=2):
                    intro = gr.Markdown(get_intro_text())
                with gr.Column(scale=1):
                    compute_location.render()
            input.render()
            downloading.render()
            extracting.render()
        with gr.Column(scale=1):
            # Add information about the system
            with gr.Accordion("ℹ️ About This App", open=True):
                ## Get number of emails (unique doc ids) in vector database
                # collection = get_collection(compute_location.value)
                # n_emails = len(set([m["doc_id"] for m in collection["metadatas"]]))
                # gr.Markdown(
                #    f"""
                #    - **Database**: *n_emails* emails from the [R-help mailing list archives](https://stat.ethz.ch/pipermail/r-help/)
                #    - **System**: retrieval and citation tools; system prompt has today's date
                #    - **Retrieval**: hybrid of dense (vector embeddings) and sparse ([BM25S](https://github.com/xhluca/bm25s))
                #    """
                # )
                info = gr.Markdown(get_info_text(compute_location.value))
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
                info="Tip: Look for 'Tool Call' and 'Next Email' separators. Quoted lines (starting with '>') are removed before indexing.",
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
    # App functions
    # -------------

    def value(value):
        """Return updated value for a component"""
        return gr.update(value=value)

    def set_compute(compute_location):
        global COMPUTE
        COMPUTE = compute_location

    def set_avatar(compute_location):
        if compute_location.startswith("cloud"):
            image_file = "images/cloud.png"
        if compute_location.startswith("edge"):
            image_file = "images/chip.png"
        return gr.update(
            avatar_images=(
                None,
                image_file,
            ),
        )

    def change_visibility(visible):
        """Return updated visibility state for a component"""
        return gr.update(visible=visible)

    def update_emails(retrieved_emails, emails_textbox):
        if retrieved_emails is None:
            # This keeps the content of the textbox when the answer step occurs
            return emails_textbox, change_visibility(True)
        elif not retrieved_emails == []:
            # This adds the retrieved emails to the textbox
            return retrieved_emails, change_visibility(True)
        else:
            # This blanks out the textbox when a new chat is started
            return "", change_visibility(False)

    def update_citations(citations):
        if citations == []:
            # Blank out and hide the citations textbox when new input is submitted
            return "", change_visibility(False)
        else:
            return citations, change_visibility(True)

    # --------------
    # Event handlers
    # --------------

    # Start a new thread when the user presses the clear (trash) button
    # https://github.com/gradio-app/gradio/issues/9722
    chatbot.clear(generate_thread_id, outputs=[thread_id], api_name=False)

    compute_location.change(
        # Update global COMPUTE variable
        set_compute,
        [compute_location],
        api_name=False,
    ).then(
        # Change the info text
        get_info_text,
        [compute_location],
        [info],
        api_name=False,
    ).then(
        # Change the chatbot avatar
        set_avatar,
        [compute_location],
        [chatbot],
        api_name=False,
    )

    show_examples.change(
        # Show examples
        change_visibility,
        [show_examples],
        [examples],
        api_name=False,
    )

    input.submit(
        # Submit input to the chatbot
        to_workflow,
        [chatbot, input, thread_id],
        [chatbot, retrieved_emails, citations_text],
        api_name=False,
    )

    retrieved_emails.change(
        # Update the emails textbox
        update_emails,
        [retrieved_emails, emails_textbox],
        [emails_textbox, emails_textbox],
        api_name=False,
    )

    citations_text.change(
        # Update the citations textbox
        update_citations,
        [citations_text],
        [citations_textbox, citations_textbox],
        api_name=False,
    )

    # ------------
    # Data loading
    # ------------

    def is_data_present():
        """Check if the database directory is present"""

        return os.path.isdir(db_dir)

    def is_data_missing():
        """Check if the database directory is missing"""

        return not os.path.isdir(db_dir)

    def download():
        """Download the db.zip file"""

        if not os.path.isdir(db_dir):

            # Define the repository URL and the directory to clone into
            url = "https://huggingface.co/datasets/jedick/R-help-db"
            to_path = "./R-help-db"

            # Clone the repository
            try:
                Repo.clone_from(url, to_path)
                print(f"Repository cloned successfully into {to_path}")
            except Exception as e:
                print(f"An error occurred while cloning {url}: {e}")

        return None

    def extract():
        """Extract the db.zip file"""

        if not os.path.isdir(db_dir):

            zip_file_path = "./R-help-db/db.zip"
            extract_to_path = "./"
            with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
                zip_ref.extractall(extract_to_path)

        return None

    false = gr.State(False)
    true = gr.State(True)
    need_data = gr.State()
    have_data = gr.State()

    # fmt: off
    demo.load(
        is_data_missing, None, [need_data], api_name=False
    ).then(
        is_data_present, None, [have_data], api_name=False
    ).then(
        change_visibility, [have_data], [input], api_name=False
    ).then(
        change_visibility, [need_data], [downloading], api_name=False
    ).then(
        download, None, [downloading], api_name=False
    ).then(
        change_visibility, [false], [downloading], api_name=False
    ).then(
        change_visibility, [need_data], [extracting], api_name=False
    ).then(
        extract, None, [extracting], api_name=False
    ).then(
        change_visibility, [false], [extracting], api_name=False
    ).then(
        change_visibility, [true], [input], api_name=False
    )
    # fmt: on


if __name__ == "__main__":

    # Set allowed_paths to serve chatbot avatar images
    current_directory = os.getcwd()
    allowed_paths = [current_directory + "/images"]
    # Launch the Gradio app
    demo.launch(allowed_paths=allowed_paths)
