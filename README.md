# R-help-chat

Mailing list subscribers often need a search facility to pull up old messages related to a topic of interest.
The R-help mailing list is no exception.
[The R-help Archives](https://stat.ethz.ch/pipermail/r-help/) are publicly available, but search tools are limited and inconsistently maintained (see the [R-help info page](https://stat.ethz.ch/mailman/listinfo/r-help)).
General-purpose AI tools like ChatGPT have knowledge of public mailing lists.
However, list users could benefit from more focused AI-powered search and chat with up-to-date list indexing.

R-help-chat is a chatbot for the R-help archives.
The end-to-end scope of this project includes data processing, email retrieval, conversational RAG, model evaluation, and deployment with a user-friendly web interface.
Domain-specific features for mailing list chatbots, like providing source citations and retrieving whole emails for context, are also included.
Here's a picture of the chatbot workflow representing a single turn (follow-up questions allow for more in-depth conversations):

![R-help-chat workflow](images/graph_LR.png)

## Features

- Complete retrieval-augmented generation (RAG) solution built with [LangChain](https://github.com/langchain-ai/langchain)
    - Chain app for simple retrieval and chat sequence
    - Graph app for conversational chat
- Data preprocesssing for email messages
    - Removes quoted lines (starting with ">") for faster indexing and retrieval
- Efficient handling for incremental data updates
    - Only indexes changed files
    - Removes stale documents from vector database
- Multiple retrieval methods for deeper search
    - Dense search with vector embeddings ([Chroma](https://github.com/chroma-core/chroma) vector database)
    - Sparse search ([BM25S](https://github.com/xhluca/bm25s))
    - Hybrid (dense+sparse) search
- Full-context retrieval
    - Each retrieval method provides whole emails (parent documents) for context
    - Dense embedding uses small chunks (child documents) to capture semantic meaning
- Tool calling and citations (with graph app)
    - [Query analysis](https://python.langchain.com/docs/tutorials/qa_chat_history/): Chat model rewrites user's query for retrieval function
    - [Source citations](https://python.langchain.com/docs/how_to/qa_sources/): Model response is structured to cite the sender and date for each answer
- Options for cloud or edge computing to balance performance, price, and privacy
    - Cloud (remote) models: OpenAI API for embeddings and chat model
    - Edge (local) models (*experimental*): [Nomic](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) embeddings and [SmolLM3](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) chat model
        - Current implementation has relatively low groundedness and accuracy scores

Planned features:

- Memory for graph app (multi-turn chat)
- Deployment on Hugging Face Spaces
- Automatic daily updates
- Post-deployment monitoring (model performance and data drift)

## Web Interface

A Gradio web interface is available for easy interaction with the chatbot:

```sh
python app.py
```

This launches an interactive chat interface in a web browser.
The interface comes with example questions and allows choosing cloud or edge processing, chain or graph app, and the retrieval strategy.

## Command-Line Usage

Setup:

- Grab one or more gzip'd files from [The R-help Archive](https://stat.ethz.ch/pipermail/r-help/)
- Extract the files and put them in a folder named `R-help`
- Choose cloud or edge models with the `compute_location` variable in `main.py`
- If using cloud processing, set your `OPENAI_API_KEY` environment variable

Run this Python code to create the vector database:

```python
from main import *
ProcessDirectory("R-help")
```

Now you're ready to run the chain or graph. Here are some examples of RAG with the chain app:

```python
RunChain("How can I get a named argument from '...'?")
# 'To get a named argument from \'...\', you can use several approaches as discussed in the context. Here are a few methods ...'

RunChain("Help with parsing REST API response.")
# 'The context provides information about parsing a REST API response in JSON format using R. Specifically, it mentions that the response from the API endpoint is in JSON format and suggests using the `jsonlite` package to parse it. ...'
```

Use the graph app to allow the chat model to rewrite your query for retrieval and to return the retrieved emails and cited sources.
In this example, the chat model cited 2 out of 6 emails retrieved for the query.

```python
# The console output shows the AI-rewritten query: print line numbers errors
result = RunGraph("How to print line numbers where errors occur?")

result["answer"]
# 'To print line numbers where errors occur in R, you can use the option `options(show.error.locations = TRUE)`. ...'

len(result["retrieved_emails"])
# 5 

result["sources"]
# 'Duncan Murdoch, 2025-01-20; Jeff Newmiller, 2025-01-18'
```

To run evals:

- Set `app_type` to graph or chain
- Set `search_type` to dense, sparse, or hybrid

```sh
python eval.py --compute_location cloud --app_type graph --search_type hybrid
```

For a fair comparison of different search types, each one retrieves up to 6 emails:

- `dense` or `sparse` (6)
- `hybrid` = `dense` + `sparse` (3 + 3)

## Evaluations

Evals are made for the following LLM-based metrics (see [NVIDIA Metrics in Ragas](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/nvidia_metrics/) for details) with OpenAI's `gpt-4o-mini` as the judge:

- **Context relevance:** degree to which retrieved context is relevant to the user query
- **Response groundedness:** how well a response is supported by the retrieved context
- **Answer accuracy:** agreement betwen the response and a reference answer

Results for queries and reference answers in `eval.csv` with retrieval from 2.5 years of the R-help archives (January 2022-June 2025):

| Compute | App | Relevance | Groundedness | Accuracy |
|-|-|-|-|-|
| Cloud | Chain | **0.80** | **0.78** | **0.72** |
| Cloud | Graph | 0.62 | 0.60     | 0.65 |
| Edge  | Chain | 0.67 | 0.60     | 0.40 |
| Edge  | Graph | 0.52 | 0.57     | 0.10 |

## Acknowledgments

This project wouldn't be what it is without these awesome codes. Thank you!

- The retriever class for BM25S (with persistence!) is copied from a [LangChain PR](https://github.com/langchain-ai/langchain/pull/28123) by [@mspronesti](https://github.com/mspronesti)
- Code from [ToolCallingLLM](https://github.com/lalanikarim/tool_calling_llm) (modified here as `tcl.py`) adds LangChain-compatible tooling to local Hugging Face models
