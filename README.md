# R-help-chat

Chat with R-help archives using an LLM. A complete RAG solution built with [LangChain](https://github.com/langchain-ai/langchain).

## Features

- Data preprocesssing for email messages
    - Removes quoted lines (starting with ">") for faster indexing and retrieval
- Efficient handling for incremental data updates
    - Only indexes changed files
    - Removes stale documents from vector database
- Hybrid retrieval combining:
    - Dense search with vector embeddings ([Chroma](https://github.com/chroma-core/chroma) vector database)
    - Sparse search ([BM25S](https://github.com/xhluca/bm25s))
    - Sparse search with reranking ([FlashRank](https://github.com/PrithivirajDamodaran/FlashRank))
- Context engineering and source tracking
    - Dense embedding uses small chunks to capture semantic meaning
    - All retrieval methods provide whole emails for context
    - *Graph app uses structured LLM response to cite the sender and date*
- Options for remote or local processing to balance performance, price, and privacy
    - Remote processing: OpenAI API for embedding and LLM
    - Local processing: [Nomic](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) embedding and [Gemma](https://huggingface.co/google/gemma-3-4b-it) LLM

## Usage

- Grab one or more gzip'd files from [The R-help Archive](https://stat.ethz.ch/pipermail/r-help/)
- Extract the files and put them in a folder named `R-help`
- Choose remote or local processing with the `embedding_type` and `chat_type` variables in `main.py`
  - If using remote processing, set your `OPENAI_API_KEY` environment variable
- Run this Python code to create the vector database:

```python
from main import *
ProcessDirectory("R-help")
```

- Now you're ready to query the database. Here are some examples:

```python
RunChain("How can I get a named argument from '...'?")
# 'To get a named argument from \'...\', you can use several approaches as discussed in the context. Here are a few methods ...'

RunChain("Help with parsing REST API response.")
# 'The context provides information about parsing a REST API response in JSON format using R. Specifically, it mentions that the response from the API endpoint is in JSON format and suggests using the `jsonlite` package to parse it. ...'
```

- Use the graph app to get source citations:

```python
RunGraph("How to print line numbers where errors occur?")
# {'answer': 'To print line numbers where errors occur in R, you can use the `options()` function to set `show.error.locations` to `TRUE`. ...',
# 'sources': ['Ivo Welch, Jan 18 2025',
#  'Luke Tierney, Jan 19 2025',
#  'Duncan Murdoch, Jan 19 2025']}
```

- To run evals:
  - Set `app_type` to graph or chain
  - Set `search_type` to dense, sparse, sparse\_rr, hybrid, or hybrid\_rr


```sh
python rag_eval.py --app_type graph --search_type hybrid_rr
```

## Evaluations

Evals are made for the following LLM-based metrics (see [available metrics in Ragas](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/) for details):

- **Context precision (CP):** proportion of retrieved chunks judged to be relevant to *reference answer*
- **Context recall (CR):** proportion of claims in *reference answer* judged to be supported by the retrieved context
- **Faithfulness (FF):** proportion of claims in *response* judged to be supported by retrieved context
- **Factual correctness (FC):** extent to which *response* aligns with *reference answer* (F1 score over atomic claims)

Results for reference answers in `rag_answers.csv` with retrieval from one month of the R-help archives (January 2025) using remote processing (OpenAI API):

| App | Search type | CP | CR | FF | FC |
|-|-|-|-|-|-|
| Chain | `hybrid`    | **0.62** | 0.74     | 0.81     | **0.72** |
| Chain | `hybrid_rr` | 0.58     | 0.77     | 0.71     | 0.69     |
| Graph | `hybrid`    | 0.55     | **0.88** | 0.85     | 0.66     |
| Graph | `hybrid_rr` | 0.61     | 0.86     | **0.92** | 0.51     |

For a fair comparison, all search types retrieve up to 6 emails that are passed to the LLM

- `hybrid` = `dense` + `sparse` (3 + 3)
- `hybrid_rr` = `dense` + `sparse` + `sparse_rr` (2 + 2 + 2)
  - `sparse_rr` is sparse search with reranking

## Acknowledgments

- The BM25S retriever code (with persistence!) is based on a [LangChain PR](https://github.com/langchain-ai/langchain/pull/28123) by [@mspronesti](https://github.com/mspronesti)
