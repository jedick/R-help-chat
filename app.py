from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from datetime import datetime
import gradio as gr
import uuid
import ast
import os
import re

# Local modules
from util import get_sources, get_start_end_months
from data import download_data, extract_data
from main import openai_model
from graph import BuildGraph

# Set environment variables
load_dotenv(dotenv_path=".env", override=True)
# Hide BM25S progress bars
os.environ["DISABLE_TQDM"] = "true"

# Database directory
db_dir = "db"

# Download and extract data if data directory is not present
if not os.path.isdir(db_dir):
    print("Downloading data ... ", end="")
    download_data()
    print("done!")
    print("Extracting data ... ", end="")
    extract_data()
    print("done!")

# Global setting for search type
search_type = "hybrid"

# Global variable for LangChain graph
# Use dictionary to store user-specific instances
# https://www.gradio.app/guides/state-in-blocks
graph_instances = {}


def delete_graph(request: gr.Request):
    """Delete a graph when the session is finished"""
    timestamp = datetime.now().replace(microsecond=0).isoformat()
    # Get session hash
    session_hash = request.session_hash
    if session_hash in graph_instances:
        del graph_instances[session_hash]
        print(f"{timestamp} - Delete graph for session {session_hash}")


def extract_think(content):
    # Added by Cursor 20250726 jmd
    # Extract content within <think>...</think>
    think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
    think_text = think_match.group(1).strip() if think_match else ""
    # Extract text after </think>
    if think_match:
        post_think = content[think_match.end() :].lstrip()
    else:
        # Check if content starts with <think> but missing closing tag
        if content.strip().startswith("<think>"):
            # Extract everything after <think>
            think_start = content.find("<think>") + len("<think>")
            think_text = content[think_start:].strip()
            post_think = ""
        else:
            # No <think> found, so return entire content as post_think
            post_think = content
    return think_text, post_think


def append_content(chunk_messages, history, thinking_about):
    """Append thinking and non-thinking content to chatbot history"""
    if chunk_messages.content:
        think_text, post_think = extract_think(chunk_messages.content)
        # Show thinking content in "metadata" message
        if think_text:
            history.append(
                gr.ChatMessage(
                    role="assistant",
                    content=think_text,
                    metadata={"title": f"🧠 Thinking about the {thinking_about}"},
                )
            )
            if not post_think and not chunk_messages.tool_calls:
                gr.Warning("Response may be incomplete", title="Thinking-only response")
        # Display non-thinking content
        if post_think:
            history.append(gr.ChatMessage(role="assistant", content=post_think))
    return history


def watchdog():
    """
    Checks for the outer watchdog file.
    If it exists, then checks for the inner watchdog file.
    If both the outer and inner watchdog files exist, then issues a warning and cleans up both files.
    If only the outer watchdog file exists, then creates the inner watchdog file.

    The purpose of the inner watchdog file is to make sure that the outer
    watchdog file exists for two consecutive ticks before issuing a warning.
    """
    outer_watchdog = "/tmp/outer_watchdog"
    inner_watchdog = "/tmp/inner_watchdog"
    if os.path.exists(outer_watchdog):
        if os.path.exists(inner_watchdog):
            message = "Building the LangGraph graph is taking longer than expected. The space may need to be restarted. Please contact the maintainer."
            gr.Warning(message, duration=None)
            try:
                os.remove(inner_watchdog)
                os.remove(outer_watchdog)
            except:
                pass
        else:
            with open(inner_watchdog, "w") as f:
                f.write("")


def run_workflow(input, collection, history, thread_id, request: gr.Request):
    """The main function to run the chat workflow"""

    # Create the outer watchdog file
    outer_watchdog = "/tmp/outer_watchdog"
    with open(outer_watchdog, "w") as f:
        f.write("")

    # Uncomment for debugging
    # print(f"Using thread_id: {thread_id}")

    # Get session hash
    session_hash = request.session_hash
    # Get graph instance if it exists
    graph = graph_instances.get(session_hash)

    if graph is None:
        # Instantiate the chat model and build the graph
        chat_model = ChatOpenAI(model=openai_model, temperature=0)
        graph_builder = BuildGraph(
            chat_model,
            db_dir,
            collection,
            search_type,
        )
        # Compile the graph with an in-memory checkpointer
        memory = MemorySaver()
        graph = graph_builder.compile(checkpointer=memory)
        # Assign graph to session
        graph_instances[session_hash] = graph
        # ISO 8601 timestamp with local timezone information without microsecond
        timestamp = datetime.now().replace(microsecond=0).isoformat()
        print(f"{timestamp} - Set {collection} graph for session {session_hash}")
    else:
        timestamp = datetime.now().replace(microsecond=0).isoformat()
        print(f"{timestamp} - Get graph for session {session_hash}")

    # Clean up the watchdog file
    try:
        os.remove(outer_watchdog)
    except:
        pass

    # Display the user input in the chatbot
    history.append(gr.ChatMessage(role="user", content=input))
    # Return the message history and empty lists for emails and citations texboxes
    yield history, [], []

    # Stream graph steps for a single input
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
            # Append thinking and non-thinking messages (if present)
            history = append_content(chunk_messages, history, thinking_about="query")
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
                        if start_year == end_year:
                            content = f"{content} ({start_year or ''})"
                        else:
                            content = (
                                f"{content} ({start_year or ''} - {end_year or ''})"
                            )
                    if "months" in args:
                        months_text = ", ".join(args["months"])
                        content = f"{content} {months_text}"
                    history.append(
                        gr.ChatMessage(
                            role="assistant",
                            content=content,
                            metadata={"title": f"🔍 Running tool {tool_call['name']}"},
                        )
                    )
            yield history, [], []

        if node == "retrieve_emails":
            chunk_messages = chunk["messages"]
            # Loop over tool calls
            count = 0
            retrieved_emails = []
            for message in chunk_messages:
                count += 1
                # Get the retrieved emails as a list
                email_list = message.content.replace(
                    "### Retrieved Emails:\n\n", ""
                ).split("--- --- --- --- Next Email --- --- --- ---\n\n")
                # If no emails were retrieved, the heading omits the colon (from graph.py)
                if email_list[0] == "### Retrieved Emails":
                    title = "❌ Retrieved 0 emails"
                    month_text = ""
                else:
                    # Get the source file names (e.g. 2024-December.txt) for retrieved emails
                    month_list = [
                        os.path.basename(text.splitlines()[0]) for text in email_list
                    ]
                    # Format months (e.g. 2024-December) into text
                    month_text = ", ".join(month_list).replace(".txt", "")
                    # Get the number of retrieved emails
                    n_emails = len(email_list)
                    title = f"🗎 Retrieved {n_emails} emails"
                history.append(
                    gr.ChatMessage(
                        role="assistant",
                        content=month_text,
                        metadata={"title": title},
                    )
                )
                # Format the retrieved emails with Tool Call heading
                retrieved_emails.append(
                    message.content.replace(
                        "### Retrieved Emails:\n\n",
                        f"### ### ### ### Tool Call {count} ### ### ### ###\n\n",
                    )
                )
            # Combine all the Tool Call results
            retrieved_emails = "\n\n".join(retrieved_emails)
            yield history, retrieved_emails, []

        if node == "answer":
            # Append messages (thinking and non-thinking) to history
            chunk_messages = chunk["messages"]
            history = append_content(chunk_messages, history, thinking_about="answer")
            # None is used for no change to the retrieved emails textbox
            yield history, None, []

        if node == "answer_with_citations":
            # Parse the message for the answer and citations
            chunk_messages = chunk["messages"][0]
            try:
                answer, citations = ast.literal_eval(chunk_messages.content)
            except:
                # In case we got an answer without citations
                answer = chunk_messages.content
                citations = None

            history.append(gr.ChatMessage(role="assistant", content=answer))
            yield history, None, citations


# Set allowed_paths to serve chatbot avatar images
current_directory = os.getcwd()
allowed_paths = [current_directory + "/images"]
# Noto Color Emoji gets a nice-looking Unicode Character “🇷” (U+1F1F7) on Chrome
theme = gr.themes.Soft(
    font=[
        "ui-sans-serif",
        "system-ui",
        "sans-serif",
        "Apple Color Emoji",
        "Segoe UI Emoji",
        "Segoe UI Symbol",
        "Noto Color Emoji",
    ]
)
# Custom CSS for bottom alignment
css = """
.row-container {
    display: flex;
    align-items: flex-end; /* Align components at the bottom */
    gap: 10px; /* Add spacing between components */
}
"""
# HTML for Font Awesome
# https://cdnjs.com/libraries/font-awesome
head = '<link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/7.0.1/css/all.min.css" rel="stylesheet">'


with gr.Blocks(
    title="R-help-chat",
    theme=theme,
    css=css,
    head=head,
) as demo:

    # -----------------
    # Define components
    # -----------------

    loading_data = gr.Textbox(
        "Please wait for the email database to be downloaded and extracted.",
        max_lines=0,
        label="Loading Data",
        visible=False,
        render=False,
    )
    downloading = gr.Textbox(
        max_lines=1,
        label="Downloading Data",
        visible=False,
        render=False,
    )
    extracting = gr.Textbox(
        max_lines=1,
        label="Extracting Data",
        visible=False,
        render=False,
    )
    missing_data = gr.Textbox(
        value="Email database is missing. Try reloading this page. If the problem persists, please contact the maintainer.",
        lines=1,
        label="Error downloading or extracting data",
        visible=False,
        render=False,
    )
    chatbot = gr.Chatbot(
        show_label=False,
        type="messages",  # Gradio 5
        # buttons=["copy_all"], # Gradio 6
        avatar_images=(None, "images/cloud.png"),
        render=False,
    )
    # Modified from gradio/chat_interface.py
    input = gr.Textbox(
        show_label=False,
        label="Message",
        placeholder="Type a message...",
        scale=7,
        autofocus=True,
        submit_btn=True,
        render=False,
    )
    emails_textbox = gr.Textbox(
        label="Retrieved Emails",
        info="Tip: Look for 'Tool Call' and 'Next Email' separators. Quoted lines (starting with '>') are removed before indexing.",
        lines=10,
        visible=False,
        render=False,
    )
    citations_textbox = gr.Textbox(
        label="Citations",
        lines=2,
        visible=False,
        render=False,
    )

    # ------------
    # Set up state
    # ------------

    def generate_thread_id():
        """Generate a new thread ID"""
        thread_id = uuid.uuid4()
        # print(f"Generated thread_id: {thread_id}")
        return thread_id

    # Define thread_id variable
    thread_id = gr.State(generate_thread_id())

    # Define states for the output textboxes
    retrieved_emails = gr.State([])
    citations_text = gr.State([])

    # ------------------
    # Make the interface
    # ------------------

    def get_intro_text():
        intro = f"""<!-- # 🤖 R-help-chat -->
            <!-- Get AI-powered answers about R programming backed by email retrieval. -->
            ## 🇷🤝💬 R-help-chat
            
            **Search and chat with the [R-help](https://stat.ethz.ch/pipermail/r-help/) and [R-devel](https://stat.ethz.ch/pipermail/r-devel/)
            mailing list archives.**
            An LLM turns your question into a search query, including year ranges and months.
            Retrieved emails are shown below the chatbot and are used by the LLM to generate an answer.
            You can ask follow-up questions with the chat history as context; changing the mailing list maintains history.
            Press the clear button (🗑) to clear the history and start a new chat.
            *Privacy notice*: Inputs and outputs are shared with OpenAI.
            """
        return intro

    def get_info_text(collection):
        try:
            # Get source files for each email and start and end months from database
            sources = get_sources(db_dir, collection)
            start, end = get_start_end_months(sources)
        except:
            # If database isn't ready, put in empty values
            sources = []
            start = None
            end = None
        info_text = f"""
            **Database:** {len(sources)} emails from {start} to {end}<br>
            **Features:** RAG, today's date, hybrid search (semantic + lexical), multiple retrievals, citations output, chat memory<br>
            **Tech:** [OpenAI](https://openai.com/), [Chroma](https://www.trychroma.com/), [BM25S](https://github.com/xhluca/bm25s),
              [Amazon S3](https://aws.amazon.com/s3/), [LangGraph](https://www.langchain.com/langgraph), [Gradio](https://www.gradio.app/)<br>
            **Maintainer:** [Jeffrey Dick](https://jedick.github.io) - feedback welcome!<br>
            **More info:** <i class="fa-brands fa-github"></i> [GitHub repository](https://github.com/jedick/R-help-chat),
              <i class="fa-brands fa-youtube"></i> [Walkthrough video](https://youtu.be/mLQqW7zea-k)
            """
        return info_text

    def get_example_questions(as_dataset=False):
        """Get example questions"""
        questions = [
            # "What is today's date?",
            "Show me code examples using plotmath",
            "When was the native pipe operator introduced?",
        ]

        # cf. https://github.com/gradio-app/gradio/pull/8745 for updating examples
        return gr.Dataset(samples=[[q] for q in questions]) if as_dataset else questions

    def get_multi_tool_questions(as_dataset=False):
        """Get multi-tool example questions"""
        questions = [
            "Differences between lapply and for loops",
            "Summarize emails from the most recent two months",
        ]

        return gr.Dataset(samples=[[q] for q in questions]) if as_dataset else questions

    def get_multi_turn_questions(as_dataset=False):
        """Get multi-turn example questions"""
        questions = [
            "Lookup emails that reference bugs.r-project.org in 2025",
            "Did the authors you cited report bugs before 2025?",
        ]

        return gr.Dataset(samples=[[q] for q in questions]) if as_dataset else questions

    def get_month_questions(as_dataset=False):
        """Get month example questions"""
        questions = [
            "Was there any discussion of ggplot2 in Q4 2025?",
            "How about Q3?",
        ]

        return gr.Dataset(samples=[[q] for q in questions]) if as_dataset else questions

    with gr.Row():
        # Left column: Intro, Compute, Chat
        with gr.Column(scale=2):
            with gr.Row(elem_classes=["row-container"]):
                with gr.Column(scale=4):
                    intro = gr.Markdown(get_intro_text())
                with gr.Column(scale=1):
                    collection = gr.Radio(
                        ["R-help", "R-devel"],
                        value="R-help",
                        label="Mailing List",
                    )
            with gr.Group() as chat_interface:
                chatbot.render()
                input.render()
            # Render textboxes for data loading progress
            loading_data.render()
            downloading.render()
            extracting.render()
            missing_data.render()
        # Right column: Info, Examples
        with gr.Column(scale=1):
            with gr.Accordion("ℹ️ App Info", open=True):
                app_info = gr.Markdown(get_info_text(collection.value))
            with gr.Accordion("💡 Examples", open=True):
                # Add some helpful examples
                example_questions = gr.Examples(
                    examples=get_example_questions(),
                    inputs=[input],
                    label="Basic examples (try R-devel for #2)",
                )
                multi_tool_questions = gr.Examples(
                    examples=get_multi_tool_questions(),
                    inputs=[input],
                    label="Multiple retrievals",
                )
                multi_turn_questions = gr.Examples(
                    examples=get_multi_turn_questions(),
                    inputs=[input],
                    label="Follow-up questions",
                )
                month_questions = gr.Examples(
                    examples=get_month_questions(),
                    inputs=[input],
                    label="Three-month periods",
                )

    # Bottom row: retrieved emails and citations
    with gr.Row():
        with gr.Column(scale=2):
            emails_textbox.render()
        with gr.Column(scale=1):
            citations_textbox.render()

    # Invisible component: timer ticking at 5-second intervals
    timer = gr.Timer(5)

    # -------------
    # App functions
    # -------------

    def value(value):
        """Return updated value for a component"""
        return gr.update(value=value)

    def change_visibility(visible):
        """Return updated visibility state for a component"""
        return gr.update(visible=visible)

    def update_textbox(content, textbox):
        if content is None:
            # Keep the content of the textbox unchanged
            return textbox, change_visibility(True)
        elif content == []:
            # Blank out the textbox
            return "", change_visibility(False)
        else:
            # Display the content in the textbox
            return content, change_visibility(True)

    # --------------
    # Event handlers
    # --------------

    # Start a new thread when the user presses the clear (trash) button
    # https://github.com/gradio-app/gradio/issues/9722
    chatbot.clear(generate_thread_id, outputs=[thread_id], api_name=False)

    collection.change(
        # We need to build a new graph if the collection changes
        delete_graph
    ).then(
        # Update the database stats in the app info box
        get_info_text,
        [collection],
        [app_info],
        api_name=False,
    )

    input.submit(
        # Submit input to the chatbot
        run_workflow,
        [input, collection, chatbot, thread_id],
        [chatbot, retrieved_emails, citations_text],
        api_name=False,
    )

    retrieved_emails.change(
        # Update the emails textbox
        update_textbox,
        [retrieved_emails, emails_textbox],
        [emails_textbox, emails_textbox],
        api_name=False,
    )

    citations_text.change(
        # Update the citations textbox
        update_textbox,
        [citations_text, citations_textbox],
        [citations_textbox, citations_textbox],
        api_name=False,
    )

    chatbot.clear(
        # Focus textbox when the chatbot is cleared
        lambda x: gr.update(value=x),
        [input],
        [input],
        api_name=False,
    )

    # Delete graph instances when page is closed/refreshed
    demo.unload(delete_graph)

    # Watch for stalled graph building
    timer.tick(watchdog)


if __name__ == "__main__":

    # Launch the Gradio app
    demo.launch(
        allowed_paths=allowed_paths,
        show_api=False,
    )
