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

Processing `2025-January.txt` (476K unzipped) takes about 30 seconds and uses 160K input tokens.

## Sample queries

```python
QueryDatabase("How can I get a named argument from '...'?")
# 'To get a named argument from \'...\', you can use several approaches as discussed in the context. Here are a few methods ...'
QueryDatabase("Help with parsing REST API response.")
# 'The context provides information about parsing a REST API response in JSON format using R. Specifically, it mentions that the response from the API endpoint is in JSON format and suggests using the `jsonlite` package to parse it. ...'
```

## Evaluations

- Evals are implemented with [Ragas](https://github.com/explodinggradients/ragas)
- The human-curated reference answers in `rag_answers.csv` are based on one month of the R-help archives (`2025-January.txt`)
- Running evals for 12 answers takes about 2.5 minutes and uses 380K input tokens

```python
python rag_eval.py
# {'context_recall': 0.5833, 'faithfulness': 0.7917, 'factual_correctness(mode=f1)': 0.7125}
```
