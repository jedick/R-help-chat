# R-help-chat

Chat with R-help archives using an LLM. A custom RAG solution built with [LangChain](https://www.langchain.com/).

## Features

- Database management to efficiently handle rolling data updates
  - Only scans changed files and removes stale documents from vector database
- Vector search on small chunks, which are then used for retrieval of whole emails
  - Embedding small chunks better captures semantic meaning
  - However, we want to retrieve the entire email for context, e.g. the date and sender
  - Uses LangChain's `ParentDocumentRetriever` and `LocalFileStore`

## Usage

- Set your `OPENAI_API_KEY` environment variable
- Grab one or more gzip'd text files from [The R-help Archive](https://stat.ethz.ch/pipermail/r-help/) and put them in a folder named `R-help`
- Run this Python code to create the vector database:

```python
from main import *
ProcessDirectory("R-help")
```

Processing `2025-January.txt` (476 KB unzipped) takes about 30 seconds and uses 160K input tokens.

## Sample queries

```python
QueryDatabase("How can I get a named argument from '...'?")
# 'To get a named argument from \'...\', you can use several approaches as discussed in the context. Here are a few methods ...'
QueryDatabase("Help with parsing REST API response.")
# 'The context provides information about parsing a REST API response in JSON format using R. Specifically, it mentions that the response from the API endpoint is in JSON format and suggests using the `jsonlite` package to parse it. ...'
```

## Evaluations

- Evals are implemented with [Ragas](https://github.com/explodinggradients/ragas)
- Choice of metrics (all LLM-based, see [available metrics in Ragas](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/) for detail):
  - **Context precision:** proportion of retrieved chunks judged to be relevant to *reference answer*
  - **Context entities recall:** proportion of entities in *reference answer* judged to be present in retrieved context
    - "This metric is useful in fact-based use cases, because where entities matter, we need the retrieved_contexts which cover them." - [Ragas docs](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_entities_recall/)
  - **Faithfulness:** proportion of claims in *response* judged to be supported by retrieved context
  - **Factual correctness:** extent to which *response* aligns with *reference answer*

Results for 12 reference answers in `rag_answers.csv` with retrieval from one month of the R-help archives (`2025-January.txt`):

```python
python rag_eval.py
# {'llm_context_precision_with_reference': 0.3819, 'context_entity_recall': 0.2827, 'faithfulness': 0.8056, 'factual_correctness(mode=f1)': 0.6942}
```

## Acknowledgments

- The BM25S retriever code (with persistence!) is based on a [LangChain PR](https://github.com/langchain-ai/langchain/pull/28123) by [@mspronesti](https://github.com/mspronesti)
