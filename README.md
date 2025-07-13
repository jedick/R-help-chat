# R-help-chat

Chat with R-help archives using an LLM. A complete RAG solution built with [LangChain](https://github.com/langchain-ai/langchain).

## Features

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
- Tool calling and citations with graph app
    - [Query analysis](https://python.langchain.com/docs/tutorials/qa_chat_history/): Chat model rewrites user's query for retrieval function
    - [Source citations](https://python.langchain.com/docs/how_to/qa_sources/): Model response is structured to cite the sources (sender and date) for each answer
- Options for remote or local processing to balance performance, price, and privacy
    - Remote processing: OpenAI API for embedding and LLM
    - Local processing: [Nomic](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) embedding and [Gemma](https://huggingface.co/google/gemma-3-4b-it) LLM

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

Results for reference answers in `rag_answers.csv` with retrieval from six months of the R-help archives (January-June 2025) using remote processing (OpenAI API):

| App | Search type | Relevance | Groundedness | Accuracy |
|-|-|-|-|-|
| Chain | `hybrid`    | 0.69     | 0.52     | **0.75** |
| Chain | `hybrid_rr` | 0.77     | 0.56     | 0.67     |
| Graph | `hybrid`    | **0.81** | 0.71     | 0.71     |
| Graph | `hybrid_rr` | 0.75     | **0.79** | 0.73     |

For a fair comparison, all search types retrieve up to 6 emails that are passed to the LLM

- `hybrid` = `dense` + `sparse` (3 + 3)
- `hybrid_rr` = `dense` + `sparse` + `sparse_rr` (2 + 2 + 2)
  - `sparse_rr` is sparse search with reranking

## Acknowledgments

- The BM25S retriever code (with persistence!) is based on a [LangChain PR](https://github.com/langchain-ai/langchain/pull/28123) by [@mspronesti](https://github.com/mspronesti)
