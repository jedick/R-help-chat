import sys
import os
import csv
from main import RunChain, RunGraph
from retriever import BuildRetriever
from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    AnswerAccuracy,
    ContextRelevance,
    ResponseGroundedness,
)
from langchain_openai import ChatOpenAI
import argparse
import logging
import traceback

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


def build_eval_dataset(queries, references, compute_mode, workflow, search_type):
    """Build dataset for evaluation"""
    dataset = []
    for query, reference in zip(queries, references):
        try:
            if workflow == "chain":
                print("\n\n--- Query ---")
                print(query)
                response = RunChain(query, compute_mode, search_type)
                print("--- Response ---")
                print(response)
                # Retrieve context documents for a query
                retriever = BuildRetriever(compute_mode, search_type)
                docs = retriever.invoke(query)
                retrieved_contexts = [doc.page_content for doc in docs]
            if workflow == "graph":
                result = RunGraph(query, compute_mode, search_type)
                retrieved_contexts = []
                if "retrieved_emails" in result:
                    # Remove the source files (e.g. R-help/2022-September.txt) as it confuses the evaluator
                    retrieved_contexts = [
                        "\n\nFrom" + email.split("\n\nFrom")[1]
                        for email in result["retrieved_emails"]
                    ]
                response = result["answer"]
            dataset.append(
                {
                    "user_input": query,
                    "retrieved_contexts": retrieved_contexts,
                    "response": response,
                    "reference": reference,
                }
            )
        except:
            print(f"--- Query omitted from evals due to failed generation: {query} ---")
            print(traceback.format_exc())

    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RAG retrieval and generation."
    )
    parser.add_argument(
        "--compute_mode",
        choices=["cloud", "edge"],
        required=True,
        help="Compute mode: cloud or edge.",
    )
    parser.add_argument(
        "--workflow",
        choices=["chain", "graph"],
        required=True,
        help="Workflow: chain or graph.",
    )
    parser.add_argument(
        "--search_type",
        choices=["dense", "sparse", "hybrid"],
        required=True,
        help="Search type: dense, sparse, or hybrid.",
    )
    args = parser.parse_args()
    compute_mode = args.compute_mode
    workflow = args.workflow
    search_type = args.search_type

    queries, references = load_queries_and_references("eval.csv")
    dataset = build_eval_dataset(
        queries, references, compute_mode, workflow, search_type
    )
    evaluation_dataset = EvaluationDataset.from_list(dataset)

    # Set up LLM for evaluation
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    evaluator_llm = LangchainLLMWrapper(llm)

    # Evaluate
    result = evaluate(
        dataset=evaluation_dataset,
        # NVIDIA metrics
        metrics=[ContextRelevance(), ResponseGroundedness(), AnswerAccuracy()],
        llm=evaluator_llm,
    )
    print("Evaluation Results:")
    print(result)


if __name__ == "__main__":
    main()
