import sys
import os
import csv
from main import QueryDatabase, build_retriever
from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness
from langchain_openai import ChatOpenAI


# Read queries and references from CSV
def load_queries_and_references(csv_path):
    queries = []
    references = []
    with open(csv_path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            queries.append(row["query"].strip('"'))
            references.append(row["reference"].strip('"'))
    return queries, references


# Retrieve context documents for a query
def get_retrieved_contexts(query):
    retriever = build_retriever()
    # ParentDocumentRetriever returns a list of Document objects
    # Use invoke instead of deprecated get_relevant_documents
    docs = retriever.invoke(query)
    return [doc.page_content for doc in docs]


# Build dataset for evaluation
def build_eval_dataset(queries, references):
    dataset = []
    for query, reference in zip(queries, references):
        retrieved_contexts = get_retrieved_contexts(query)
        response = QueryDatabase(query)
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
    queries, references = load_queries_and_references("rag_answers.csv")
    dataset = build_eval_dataset(queries, references)
    evaluation_dataset = EvaluationDataset.from_list(dataset)

    # Set up LLM for evaluation
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    evaluator_llm = LangchainLLMWrapper(llm)

    # Evaluate
    result = evaluate(
        dataset=evaluation_dataset,
        metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness()],
        llm=evaluator_llm,
    )
    print("Evaluation Results:")
    print(result)


if __name__ == "__main__":
    main()
