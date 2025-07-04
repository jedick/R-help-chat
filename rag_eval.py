import sys
import os
import csv
from main import QueryDatabase, build_retriever
from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    LLMContextPrecisionWithReference,
    ContextEntityRecall,
    Faithfulness,
    FactualCorrectness,
)

# NVIDIA metrics
# from ragas.metrics import AnswerAccuracy, ContextRelevance, ResponseGroundedness
from langchain_openai import ChatOpenAI
import argparse


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
    retriever = build_retriever(search_type)
    # Use invoke instead of deprecated get_relevant_documents
    docs = retriever.invoke(query)
    return [doc.page_content for doc in docs]


def build_eval_dataset(queries, references, search_type):
    """Build dataset for evaluation"""
    dataset = []
    for query, reference in zip(queries, references):
        retrieved_contexts = get_retrieved_contexts(query, search_type)
        response = QueryDatabase(query, search_type=search_type)
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
        "--search_type",
        choices=["dense", "sparse", "hybrid"],
        required=True,
        help="Retrieval type: dense, sparse, or hybrid.",
    )
    args = parser.parse_args()
    search_type = args.search_type

    queries, references = load_queries_and_references("rag_answers.csv")
    dataset = build_eval_dataset(queries, references, search_type)
    evaluation_dataset = EvaluationDataset.from_list(dataset)

    # Set up LLM for evaluation
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    evaluator_llm = LangchainLLMWrapper(llm)

    # Evaluate
    result = evaluate(
        dataset=evaluation_dataset,
        metrics=[
            LLMContextPrecisionWithReference(),
            ContextEntityRecall(),
            Faithfulness(),
            FactualCorrectness(),
        ],
        # NVIDIA metrics
        # metrics=[AnswerAccuracy(), ContextRelevance(), ResponseGroundedness()],
        llm=evaluator_llm,
    )
    print("Evaluation Results:")
    print(result)


if __name__ == "__main__":
    main()
