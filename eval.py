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


def load_questions_and_references(csv_path):
    """Read questions and references from CSV"""
    questions = []
    references = []
    with open(csv_path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            questions.append(row["question"].strip('"'))
            references.append(row["reference"].strip('"'))
    return questions, references


def build_eval_dataset(questions, references, compute_mode, workflow, search_type):
    """Build dataset for evaluation"""
    dataset = []
    for question, reference in zip(questions, references):
        try:
            if workflow == "chain":
                print("\n\n--- Question ---")
                print(question)
                response = RunChain(question, compute_mode, search_type)
                print("--- Response ---")
                print(response)
                # Retrieve context documents for a question
                retriever = BuildRetriever(compute_mode, search_type)
                docs = retriever.invoke(question)
                retrieved_contexts = [doc.page_content for doc in docs]
            if workflow == "graph":
                result = RunGraph(question, compute_mode, search_type)
                retrieved_contexts = []
                if "retrieved_emails" in result:
                    # Remove the source file names (e.g. R-help/2022-September.txt) as it confuses the evaluator
                    retrieved_contexts = [
                        "\n\n\nFrom" + email.split("\n\n\nFrom")[1]
                        for email in result["retrieved_emails"]
                    ]
                response = result["answer"]
            dataset.append(
                {
                    "user_input": question,
                    "retrieved_contexts": retrieved_contexts,
                    "response": response,
                    "reference": reference,
                }
            )
        except:
            print(
                f"--- Question omitted from evals due to failed generation: {question} ---"
            )
            print(traceback.format_exc())

    return dataset


def run_evals_with_csv(csv_path):
    """Run evals using saved responses in a CSV file"""

    # Load an evaluation dataset from saved responses in a CSV file
    csv_questions = []
    retrieved_emails = []
    answers = []

    with open(csv_path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            csv_questions.append(row["question"].strip('"'))
            retrieved_emails.append(row["retrieved_emails"].strip('"'))
            answers.append(row["answer"].strip('"'))

    questions, references = load_questions_and_references("eval.csv")

    # Make sure the questions are the same
    assert csv_questions == questions

    # Build dataset for evaluation
    dataset = []
    for question, reference, retrieved_email, answer in zip(
        questions, references, retrieved_emails, answers
    ):
        retrieved_contexts = [
            "\n\n\nFrom" + email for email in retrieved_email.split("\n\n\nFrom")
        ]
        # Remove the source file names (e.g. R-help/2022-September.txt) as it confuses the evaluator
        retrieved_contexts = [
            "\n\n\nFrom" + email.split("\n\n\nFrom")[1]
            for email in retrieved_email.split(
                "\n\n--- --- --- --- Next Email --- --- --- ---\n\n"
            )
        ]
        dataset.append(
            {
                "user_input": question,
                "retrieved_contexts": retrieved_contexts,
                "response": answer,
                "reference": reference,
            }
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


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RAG retrieval and generation."
    )
    parser.add_argument(
        "--compute_mode",
        choices=["remote", "local"],
        required=True,
        help="Compute mode: remote or local.",
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

    questions, references = load_questions_and_references("eval.csv")
    dataset = build_eval_dataset(
        questions, references, compute_mode, workflow, search_type
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
