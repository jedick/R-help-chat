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
from retriever import db_dir

# Set environment variables
load_dotenv(dotenv_path=".env", override=True)
# Hide BM25S progress bars
os.environ["DISABLE_TQDM"] = "true"

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

# Global variables for LangChain graph: use dictionaries to store user-specific instances
# https://www.gradio.app/guides/state-in-blocks
graph_instances = {}


def cleanup_graph(request: gr.Request):
    timestamp = datetime.now().replace(microsecond=0).isoformat()
    if request.session_hash in graph_instances:
        del graph_instances[request.session_hash]
        print(f"{timestamp} - Delete graph for session {request.session_hash}")


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


def run_workflow(input, history, thread_id, session_hash):
    """The main function to run the chat workflow"""

    # Get graph instance
    graph = graph_instances.get(session_hash)

    if graph is None:
        # Get the chat model and build the graph
        chat_model = ChatOpenAI(model=openai_model, temperature=0)
        graph_builder = BuildGraph(
            chat_model,
            search_type,
        )
        # Compile the graph with an in-memory checkpointer
        memory = MemorySaver()
        graph = graph_builder.compile(checkpointer=memory)
        # Set global graph
        graph_instances[session_hash] = graph
        # ISO 8601 timestamp with local timezone information without microsecond
        timestamp = datetime.now().replace(microsecond=0).isoformat()
        print(f"{timestamp} - Set graph for session {session_hash}")
        ## Notify when model finishes loading
        # gr.Success("Model loaded!", duration=4)
    else:
        timestamp = datetime.now().replace(microsecond=0).isoformat()
        print(f"{timestamp} - Get graph for session {session_hash}")

    # print(f"Using thread_id: {thread_id}")

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
                        content = f"{content} ({start_year or ''} - {end_year or ''})"
                    if "months" in args:
                        content = f"{content} {args['months']}"
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
                # Get the list of source files (e.g. R-help/2024-December.txt) for retrieved emails
                month_list = [text.splitlines()[0] for text in email_list]
                # Format months (e.g. 2024-December) into text
                month_text = (
                    ", ".join(month_list).replace("R-help/", "").replace(".txt", "")
                )
                # Get the number of retrieved emails
                n_emails = len(email_list)
                title = f"🗎 Retrieved {n_emails} emails"
                if email_list[0] == "### No emails were retrieved":
                    title = "❌ Retrieved 0 emails"
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


def to_workflow(request: gr.Request, *args):
    """Wrapper function to call run_workflow() with session_hash"""
    input = args[0]
    # Add session_hash to arguments
    new_args = args + (request.session_hash,)
    for value in run_workflow(*new_args):
        yield value


with gr.Blocks(
    title="R-help-chat",
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
        avatar_images=(None, "images/cloud.png"),
        buttons=["copy", "copy_all"],
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
            
            **Search and chat with the [R-help mailing list archives](https://stat.ethz.ch/pipermail/r-help/).**
            An LLM turns your question into a search query, including year ranges and months.
            Retrieved emails are shown below the chatbot and are used by the LLM to generate an answer.
            You can ask follow-up questions with the chat history as context.
            Press the clear button (🗑) to clear the history and start a new chat.
            *Privacy notice*: Data sharing with OpenAI is enabled.
            """
        return intro

    def get_info_text():
        try:
            # Get source files for each email and start and end months from database
            sources = get_sources()
            start, end = get_start_end_months(sources)
        except:
            # If database isn't ready, put in empty values
            sources = []
            start = None
            end = None
        info_text = f"""
            **Database:** {len(sources)} emails from {start} to {end}<br>
            **Models:** {openai_model} and text-embedding-3-small<br>
            **Features:** RAG, today's date, hybrid search (semantic + lexical), multiple retrievals, citations output, chat memory<br>
            **Tech:** [OpenAI](https://openai.com/), [Chroma](https://www.trychroma.com/),
              [BM25S](https://github.com/xhluca/bm25s), [LangGraph](https://www.langchain.com/langgraph), [Gradio](https://www.langchain.com/langgraph)<br>
            🏠 **More info:** [R-help-chat GitHub repository](https://github.com/jedick/R-help-chat)
            """
        return info_text

    def get_example_questions(as_dataset=True):
        """Get example questions"""
        questions = [
            # "What is today's date?",
            "Summarize emails from the most recent two months",
            "Show me code examples using plotmath",
            "When was has.HLC mentioned?",
            "Who reported installation problems in 2023-2024?",
        ]

        # cf. https://github.com/gradio-app/gradio/pull/8745 for updating examples
        return gr.Dataset(samples=[[q] for q in questions]) if as_dataset else questions

    def get_multi_tool_questions(as_dataset=True):
        """Get multi-tool example questions"""
        questions = [
            "Differences between lapply and for loops",
            "Discuss pipe operator usage in 2022, 2023, and 2024",
        ]

        return gr.Dataset(samples=[[q] for q in questions]) if as_dataset else questions

    def get_multi_turn_questions(as_dataset=True):
        """Get multi-turn example questions"""
        questions = [
            "Lookup emails that reference bugs.r-project.org in 2025",
            "Did the authors you cited report bugs before 2025?",
        ]

        return gr.Dataset(samples=[[q] for q in questions]) if as_dataset else questions

    with gr.Row():
        # Left column: Intro, Compute, Chat
        with gr.Column(scale=2):
            with gr.Row(elem_classes=["row-container"]):
                with gr.Column(scale=4):
                    intro = gr.Markdown(get_intro_text())
                with gr.Column(scale=1):
                    gr.Radio(
                        ["Auto", "R-help", "R-devel", "R-pkg-devel"],
                        label="🚧 Mailing List (Under construction) 🚧",
                        interactive=False,
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
                info = gr.Markdown(get_info_text())
            with gr.Accordion("💡 Examples", open=True):
                # Add some helpful examples
                example_questions = gr.Examples(
                    examples=get_example_questions(as_dataset=False),
                    inputs=[input],
                    label="Click an example to fill the message box",
                )
                multi_tool_questions = gr.Examples(
                    examples=get_multi_tool_questions(as_dataset=False),
                    inputs=[input],
                    label="Multiple retrievals",
                )
                multi_turn_questions = gr.Examples(
                    examples=get_multi_turn_questions(as_dataset=False),
                    inputs=[input],
                    label="Asking follow-up questions",
                )

    # Bottom row: retrieved emails and citations
    with gr.Row():
        with gr.Column(scale=2):
            emails_textbox.render()
        with gr.Column(scale=1):
            citations_textbox.render()

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
    chatbot.clear(generate_thread_id, outputs=[thread_id], api_visibility="private")

    input.submit(
        # Submit input to the chatbot
        to_workflow,
        [input, chatbot, thread_id],
        [chatbot, retrieved_emails, citations_text],
        api_visibility="private",
    )

    retrieved_emails.change(
        # Update the emails textbox
        update_textbox,
        [retrieved_emails, emails_textbox],
        [emails_textbox, emails_textbox],
        api_visibility="private",
    )

    citations_text.change(
        # Update the citations textbox
        update_textbox,
        [citations_text, citations_textbox],
        [citations_textbox, citations_textbox],
        api_visibility="private",
    )

    chatbot.clear(
        # Focus textbox when the chatbot is cleared
        lambda x: gr.update(value=x),
        [input],
        [input],
        api_visibility="private",
    )

    # Clean up graph instances when page is closed/refreshed
    demo.unload(cleanup_graph)


if __name__ == "__main__":

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
    # Launch the Gradio app
    demo.launch(
        allowed_paths=allowed_paths,
        theme=theme,
        css=css,
        footer_links=["gradio", "settings"],
    )
