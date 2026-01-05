from datetime import date
from util import get_sources, get_start_end_months
import re


def check_prompt(prompt):
    """Check for unassigned variables"""
    # A sanity check that we don't have unassigned variables
    matches = re.findall(r"\{.*?\}", " ".join(prompt))
    if matches:
        raise ValueError(f"Unassigned variables in prompt: {' '.join(matches)}")
    return prompt


def query_prompt(db_dir, collection):
    """
    Return system prompt for query step

    Args:
        db_dir: Database directory
        collection: Email collection
    """

    # Get start and end months from database
    start, end = get_start_end_months(get_sources(db_dir, collection))
    # Use appropriate list topic
    if collection == "R-help":
        topic = "R programming"
    elif collection == "R-devel":
        topic = "R development"
    elif collection == "R-package-devel":
        topic = "R package development"

    prompt = (
        f"Today Date: {date.today()}. "
        f"You are a search assistant for retrieving information about {topic} from the {collection} mailing list archives. "
        "Write a search query to retrieve emails relevant to the user's question. "
        "Do not answer the user's question and do not ask the user for more information. "
        # gpt-4o-mini thinks last two months aren't available with this: "Emails from from {start} to {end} are available for retrieval. "
        f"The emails available for retrieval are from {start} to {end}. "
        "For questions about differences, changes, or comparisons between X and Y, retrieve emails about X and Y using separate tool calls. "
        "Also use multiple tool calls for multiple months or years but not long year ranges (> 5 years). "
        "Always use retrieve_emails with a non-empty query string for search_query. "
        "For general summaries, use retrieve_emails(search_query='R'). "
        "For questions about years, use retrieve_emails(search_query=<query>, start_year=, end_year=). "
        "For questions about months, use 3-letter abbreviations (Jan...Dec) for the 'months' argument. "
        "Use all previous messages as context to formulate your search query. "  # Gemma
        "You should always retrieve more emails based on context and the most recent question. "  # Qwen
        f"If you decide not to retrieve emails, tell the user how to improve their question to search the {collection} mailing list. "
    )
    prompt = check_prompt(prompt)

    return prompt


def answer_prompt(collection):
    """Return system prompt for answer step"""

    # Use appropriate list topic
    if collection == "R-help":
        topic = "R programming"
    elif collection == "R-devel":
        topic = "R development"
    elif collection == "R-package-devel":
        topic = "R package development"

    prompt = (
        f"Today Date: {date.today()}. "
        f"You are a helpful chatbot that can answer questions about {topic} based on the {collection} mailing list archives. "
        "Summarize the retrieved emails to answer the user's question or query. "
        "If any of the retrieved emails are irrelevant (e.g. wrong dates), then do not use them. "
        "Tell the user if there are no retrieved emails or if you are unable to answer the question based on the information in the emails. "
        "Do not give an answer based on your own knowledge or memory, and do not include examples that aren't based on the retrieved emails. "
        "Example: For a question about using lm(), take examples of lm() from the retrieved emails to answer the user's question. "
        # "Do not respond with packages that are only listed under sessionInfo, session info, or other attached packages. "
        "You must include inline citations (email senders and dates) in each part of your response. "
        "Only answer general questions about R if the answer is in the retrieved emails. "
        "Only include URLs if they were used by human authors (not in email headers), and do not modify any URLs. "  # Qwen, Gemma
        "Respond with 500 words maximum and 50 lines of code maximum. "
        "Use answer_with_citations to provide the complete answer and all citations used. "
    )
    prompt = check_prompt(prompt)

    return prompt
