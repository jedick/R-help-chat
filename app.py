import gradio as gr
import torch
from main import RunChain, RunGraph, embedding_type, chat_type


def check_gpu_availability():
    """Check if GPU is available for local models"""
    return torch.cuda.is_available()


def chat_with_r_help(
    query: str,
    embedding_type: str,
    chat_type: str,
    app_type: str,
    search_type: str = "hybrid_rr",
):
    """
    Main chat function that handles the R-help chatbot interaction

    Args:
        query: User's question
        embedding_type: "remote" or "local" for embeddings
        chat_type: "remote" or "local" for chat model
        app_type: "chain" or "graph" for the application type
        search_type: Type of search to use (default: "hybrid_rr")

    Returns:
        str: The chatbot's response
    """

    # Validate inputs
    if not query.strip():
        return "Please enter a question."

    # Check GPU availability for local models
    if not check_gpu_availability() and (
        embedding_type == "local" or chat_type == "local"
    ):
        return "Error: Local models selected but no GPU available. Please use remote models or ensure GPU is available."

    try:
        # Set the global configuration
        import main

        main.embedding_type = embedding_type
        main.chat_type = chat_type

        # Run the appropriate function based on app_type
        if app_type == "chain":
            response = RunChain(query, search_type=search_type, chat_type=chat_type)
            return response
        elif app_type == "graph":
            result = RunGraph(
                query=query,
                search_type=search_type,
                chat_type=chat_type,
                think_retrieve=False,
                think_generate=False,
            )
            # Extract the answer from the graph result
            if isinstance(result, dict) and "answer" in result:
                return result["answer"]
            elif hasattr(result, "messages") and result.messages:
                # Get the last message content
                last_message = result.messages[-1]
                if hasattr(last_message, "content"):
                    return last_message.content
                else:
                    return str(last_message)
            else:
                return str(result)
        else:
            return (
                f"Error: Unknown app_type '{app_type}'. Please use 'chain' or 'graph'."
            )

    except Exception as e:
        return f"Error: {str(e)}"


# Create the Gradio interface
with gr.Blocks(title="R-help-chat", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🤖 R-help-chat
        
        Chat with the R-help mailing list archives using AI. Ask questions about R programming and get answers based on real discussions from the R-help community.
        
        **How to use:**
        1. Select your preferred model settings (remote/local for embeddings and chat)
        2. Choose the application type (chain or graph)
        3. Enter your R programming question
        4. Get an AI-generated answer based on R-help archives
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            # Configuration options
            gr.Markdown("### ⚙️ Configuration")

            embedding_dropdown = gr.Dropdown(
                choices=["remote", "local"],
                value="remote",
                label="Embedding Type",
                info="Remote: OpenAI API, Local: Nomic embeddings (requires GPU)",
            )

            chat_dropdown = gr.Dropdown(
                choices=["remote", "local"],
                value="remote",
                label="Chat Type",
                info="Remote: OpenAI API, Local: SmolLM3 (requires GPU)",
            )

            app_dropdown = gr.Dropdown(
                choices=["chain", "graph"],
                value="chain",
                label="Application Type",
                info="Chain: Simple RAG, Graph: Conversational RAG with sources",
            )

            search_dropdown = gr.Dropdown(
                choices=["dense", "sparse", "sparse_rr", "hybrid", "hybrid_rr"],
                value="hybrid_rr",
                label="Search Type",
                info="Different retrieval strategies",
            )

            # GPU status indicator
            gpu_status = gr.Markdown(
                f"🟢 GPU Available: {gpu_available}"
                if check_gpu_availability()
                else "🔴 GPU Not Available - Local models will not work"
            )

        with gr.Column(scale=2):
            # Chat interface
            gr.Markdown("### 💬 Chat Interface")

            query_input = gr.Textbox(
                label="Your Question",
                placeholder="Ask about R programming... (e.g., 'How do I read a CSV file in R?')",
                lines=3,
            )

            submit_btn = gr.Button("Ask Question", variant="primary", size="lg")

            response_output = gr.Textbox(
                label="AI Response", lines=10, interactive=False
            )

    # Add some helpful examples
    with gr.Accordion("💡 Example Questions", open=False):
        gr.Markdown(
            """
            Here are some example questions you can try:
            
            - "How can I get a named argument from '...'?"
            - "Help with parsing REST API response."
            - "How to print line numbers where errors occur?"
            - "What are the differences between data.frame and tibble?"
            - "How do I install packages from GitHub?"
            - "What's the best way to handle missing values in R?"
            """
        )

    # Add information about the system
    with gr.Accordion("ℹ️ About This System", open=False):
        gr.Markdown(
            """
            **R-help Chatbot** is built on the R-help mailing list archives, providing AI-powered answers to R programming questions.
            
            **Features:**
            - **Hybrid Retrieval**: Combines dense vector search and sparse BM25 search
            - **Source Citations**: Graph mode provides citations from R-help discussions
            - **Multiple Models**: Support for both remote (OpenAI) and local models
            - **Conversational RAG**: Graph mode supports multi-turn conversations
            
            **Model Options:**
            - **Remote**: Uses OpenAI API (requires API key)
            - **Local**: Uses local models (requires GPU)
            
            **Application Types:**
            - **Chain**: Simple retrieval-augmented generation
            - **Graph**: Advanced conversational RAG with tool calls and structured output
            
            **Search Types:**
            - **dense**: Vector similarity search
            - **sparse**: BM25 keyword search
            - **sparse_rr**: BM25 with reranking
            - **hybrid**: Combination of dense and sparse
            - **hybrid_rr**: Hybrid with reranking (recommended)
            """
        )

    # Connect the submit button to the chat function
    submit_btn.click(
        fn=chat_with_r_help,
        inputs=[
            query_input,
            embedding_dropdown,
            chat_dropdown,
            app_dropdown,
            search_dropdown,
        ],
        outputs=response_output,
    )

    # Also allow Enter key to submit
    query_input.submit(
        fn=chat_with_r_help,
        inputs=[
            query_input,
            embedding_dropdown,
            chat_dropdown,
            app_dropdown,
            search_dropdown,
        ],
        outputs=response_output,
    )


if __name__ == "__main__":
    # Launch the interface
    demo.launch(
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,  # Default Gradio port
        share=False,  # Set to True to create a public link
        debug=True,  # Enable debug mode for development
    )
