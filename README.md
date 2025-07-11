# R-help-chat

Chat with R-help archives using an LLM. A complete RAG solution built with [LangChain](https://www.langchain.com/).

## Features

- Data preprocesssing for email messages
    - Removes quoted lines (starting with ">") for faster indexing and retrieval
- Efficient handling for incremental data updates
    - Only indexes changed files
    - Removes stale documents from vector database
- Vector search on small chunks, which are then used for retrieval of whole emails
    - Embedding small chunks better captures semantic meaning
    - However, we want to retrieve the entire email for context, e.g. the date and sender
    - Uses LangChain's `ParentDocumentRetriever` and `LocalFileStore`
- Hybrid retrieval using ensemble of:
    - Dense search with vector embeddings ([Chroma](https://github.com/chroma-core/chroma) vector database)
    - Sparse search ([BM25S](https://github.com/xhluca/bm25s))
    - Sparse search with reranking ([FlashRank](https://github.com/PrithivirajDamodaran/FlashRank))
- Options for remote or local processing to balance performance, price, and privacy
    - Remote processing: OpenAI API for embedding and LLM
    - Local processing: [Nomic](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) embedding and [Gemma](https://huggingface.co/google/gemma-3-4b-it) LLM

## Usage

- Grab one or more gzip'd files from [The R-help Archive](https://stat.ethz.ch/pipermail/r-help/)
- Extract the files and put them in a folder named `R-help`
- Choose remote or local processing with the `embedding_api` and `llm_api` variables in `main.py`
  - If using remote processing, set your `OPENAI_API_KEY` environment variable
- Run this Python code to create the vector database:

```python
from main import *
ProcessDirectory("R-help")
```

- Now you're ready to query the database. Here are some examples:

```python
QueryDatabase("How can I get a named argument from '...'?")
# 'To get a named argument from \'...\', you can use several approaches as discussed in the context. Here are a few methods ...'
QueryDatabase("Help with parsing REST API response.")
# 'The context provides information about parsing a REST API response in JSON format using R. Specifically, it mentions that the response from the API endpoint is in JSON format and suggests using the `jsonlite` package to parse it. ...'
```

- To run evals (set search type to `dense`, `sparse`, or `hybrid`):

```sh
python rag_eval.py --search_type hybrid
```

## Evaluations

Evals are made for the following LLM-based metrics (see [available metrics in Ragas](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/) for details):

- **Context precision:** proportion of retrieved chunks judged to be relevant to *reference answer*
- **Context recall:** proportion of claims in *reference answer* judged to be supported by the retrieved context
- **Faithfulness:** proportion of claims in *response* judged to be supported by retrieved context
- **Factual correctness:** extent to which *response* aligns with *reference answer* (F1 score over atomic claims)

Results for 12 reference answers in `rag_answers.csv` with retrieval from one month of the R-help archives (January 2025):

| Processing | Search type | Context precision | Context recall | Faithfulness | Factual correctness |
|-|-|-|-|-|-|
| Remote | `dense`     | 0.59     | 0.74     | 0.77     | 0.68     |
| Remote | `sparse`    | 0.59     | 0.83     | **0.89** | 0.68     |
| Remote | `sparse_rr` | 0.49     | **0.87** | 0.67     | **0.78** |
| Remote | `hybrid`    | **0.62** | 0.74     | 0.81     | 0.72     |
| Remote | `hybrid_rr` | 0.58     | 0.77     | 0.71     | 0.69     |

For a fair comparison, all search types retrieve up to 6 emails that are passed to the LLM

- `sparse_rr` is sparse search with reranking
- `hybrid` = `dense` + `sparse` (3 + 3)
- `hybrid_rr` = `dense` + `sparse` + `sparse_rr` (2 + 2 + 2)

## Acknowledgments

- The BM25S retriever code (with persistence!) is based on a [LangChain PR](https://github.com/langchain-ai/langchain/pull/28123) by [@mspronesti](https://github.com/mspronesti)
