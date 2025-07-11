import sys
import os
import csv
from main import RunChain, RunGraph, embedding_type, chat_type
from build_retriever import BuildRetriever
from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    LLMContextPrecisionWithReference,
    ContextRecall,
    Faithfulness,
    FactualCorrectness,
)
from langchain_openai import ChatOpenAI
import argparse
import logging

# Suppress these messages:
# INFO:openai._base_client:Retrying request to /chat/completions in ___ seconds
# https://community.openai.com/t/suppress-http-request-post-message/583334/8
openai_logger = logging.getLogger("openai")
openai_logger.setLevel(logging.WARNING)


def load_queries_and_references(csv_path):
    """Read queries and references from CSV"""
    queries = []
    references = []
    with open(csv_path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            queries.append(row["query"].strip('"'))
            references.append(row["reference"].strip('"'))
    return queries, references


def get_retrieved_contexts(query, search_type):
    """Retrieve context documents for a query"""
    retriever = BuildRetriever(search_type, embedding_type)
    # Use invoke instead of deprecated get_relevant_documents
    docs = retriever.invoke(query)
    return [doc.page_content for doc in docs]


def build_eval_dataset(queries, references, app_type, search_type):
    """Build dataset for evaluation"""
    dataset = []
    for query, reference in zip(queries, references):
        retrieved_contexts = get_retrieved_contexts(query, search_type)
        if app_type == "chain":
            response = RunChain(query, search_type=search_type)
        if app_type == "graph":
            result = RunGraph(query, search_type=search_type)
            response = result["messages"][-1].content
        dataset.append(
            {
                "user_input": query,
                "retrieved_contexts": retrieved_contexts,
                "response": response,
                "reference": reference,
            }
        )
    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RAG retrieval and generation."
    )
    parser.add_argument(
        "--app_type",
        choices=["chain", "graph"],
        required=True,
        help="App type: chain or graph.",
    )
    parser.add_argument(
        "--search_type",
        choices=["dense", "sparse", "sparse_rr", "hybrid", "hybrid_rr"],
        required=True,
        help="Search type: dense, sparse, sparse_rr, hybrid, or hybrid_rr.",
    )
    args = parser.parse_args()
    app_type = args.app_type
    search_type = args.search_type

    queries, references = load_queries_and_references("rag_answers.csv")
    dataset = build_eval_dataset(queries, references, app_type, search_type)
    evaluation_dataset = EvaluationDataset.from_list(dataset)

    # Set up LLM for evaluation
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    evaluator_llm = LangchainLLMWrapper(llm)

    # Evaluate
    result = evaluate(
        dataset=evaluation_dataset,
        metrics=[
            LLMContextPrecisionWithReference(),
            ContextRecall(),
            Faithfulness(),
            FactualCorrectness(),
        ],
        # NVIDIA metrics (not used - more concise, but less explainable)
        # metrics=[AnswerAccuracy(), ContextRelevance(), ResponseGroundedness()],
        llm=evaluator_llm,
    )
    print("Evaluation Results:")
    print(result)


if __name__ == "__main__":
    main()
