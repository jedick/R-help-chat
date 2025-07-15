# R-help-chat

Mailing list subscribers often need a search facility to pull up old messages related to a topic of interest.
The R-help mailing list is no exception.
[The R-help Archives](https://stat.ethz.ch/pipermail/r-help/) are publicly available, but search tools are limited and inconsistently maintained (see the [R-help info page](https://stat.ethz.ch/mailman/listinfo/r-help)).
General-purpose AI tools like ChatGPT have knowledge of public mailing lists.
However, list users could benefit from more focused AI-powered search and chat with up-to-date list indexing.

R-help-chat is a chatbot for the R-help archives.
This project develops an end-to-end workflow including data processing, multiple retrieval methods, conversational chat, evaluation, and deployment and monitoring.
Domain-specific requirements for mailing list chats, like source citations and whole emails for chat context, are also included.

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
    - Sparse search with reranking ([FlashRank](https://github.com/PrithivirajDamodaran/FlashRank))
- Full-context retrieval
    - Each retrieval method provides whole emails (parent documents) for context
    - Dense embedding uses small chunks (child documents) to capture semantic meaning
- Tool calling and citations (with graph app)
    - [Query analysis](https://python.langchain.com/docs/tutorials/qa_chat_history/): Chat model rewrites user's query for retrieval function
    - [Source citations](https://python.langchain.com/docs/how_to/qa_sources/): Model response is structured to cite the sender and date for each answer
- Options for remote or local processing to balance performance, price, and privacy
    - Remote processing: OpenAI API for embedding and chat model
    - Local processing (*experimental*): [Nomic](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) embedding and [SmolLM3](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) chat model
        - Current implementation has relatively low groundedness and accuracy scores

Planned features:
- Memory for graph app (multi-turn chat)
- Deployment on Hugging Face Spaces
- Automatic daily updates
- Post-deployment monitoring (model performance and data drift)

## Usage

Setup:

- Grab one or more gzip'd files from [The R-help Archive](https://stat.ethz.ch/pipermail/r-help/)
- Extract the files and put them in a folder named `R-help`
- Choose remote or local processing with the `embedding_type` and `chat_type` variables in `main.py`
- If using remote processing, set your `OPENAI_API_KEY` environment variable

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

Use the graph app to get the context and cited sources.
In this example, the chat model cited 3 out of 5 emails retrieved as context for the query.

```python
result = RunGraph("How to print line numbers where errors occur?")

result["messages"][-1].content
# 'To print line numbers where errors occur in R, you can use the `options()` function to set `show.error.locations` to `TRUE`. ...',

len(result["context"])
# 5 

result["sources"]
# ['Duncan Murdoch, Sat, 18 Jan 2025',
# 'Luke Tierney, Sun, 19 Jan 2025',
# 'Duncan Murdoch, Mon, 20 Jan 2025']
```

To run evals:

- Set `app_type` to graph or chain
- Set `search_type` to dense, sparse, sparse\_rr, hybrid, or hybrid\_rr

```sh
python rag_eval.py --app_type graph --search_type hybrid_rr
```

## Evaluations

Evals are made for the following LLM-based metrics (see [NVIDIA Metrics in Ragas](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/nvidia_metrics/) for details):

- **Context relevance:** degree to which retrieved context is relevant to the user query
- **Response groundedness:** how well a response is supported by the retrieved context
- **Answer accuracy:** agreement betwen the response and a reference answer

Results for queries and reference answers in `rag_answers.csv` with retrieval from six months of the R-help archives (January-June 2025):

| Processing | App | Search type | Relevance | Groundedness | Accuracy |
|-|-|-|-|-|-|
| Remote | Chain | `hybrid`    | 0.69     | 0.52     | **0.75** |
| Remote | Chain | `hybrid_rr` | 0.77     | 0.56     | 0.67     |
| Remote | Graph | `hybrid`    | **0.81** | 0.71     | 0.71     |
| Remote | Graph | `hybrid_rr` | 0.75     | **0.79** | 0.73     |
| Local  | Graph | `hybrid`    | 0.64     | 0.54     | 0.31     |
| Local  | Graph | `hybrid_rr` | 0.80     | 0.39     | 0.27     |

For a fair comparison, all search types retrieve up to 6 emails that are passed to the chat model:

- `hybrid` = `dense` + `sparse` (3 + 3)
- `hybrid_rr` = `dense` + `sparse` + `sparse_rr` (2 + 2 + 2)
  - `sparse_rr` is sparse search with reranking

## Acknowledgments

This project wouldn't be what it is without these awesome codes. Thank you!

- The BM25S retriever code (with persistence!) is copied from a [LangChain PR](https://github.com/langchain-ai/langchain/pull/28123) by [@mspronesti](https://github.com/mspronesti)
- [ToolCallingLLM](https://github.com/lalanikarim/tool_calling_llm) is used to add LangChain-compatible tooling to a local Hugging Face model
