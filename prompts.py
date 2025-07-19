from datetime import date
from util import get_sources, get_start_end_months
import re


def retrieve_prompt(compute_location):
    """Return system prompt for query step

    Args:
        compute_location: Compute location for embedding model (cloud or edge)
    """

    # Get start and end months from database
    start, end = get_start_end_months(get_sources(compute_location))

    retrieve_prompt = (
        f"The current date is {date.today()}. "
        "You are a helpful RAG chatbot designed to answer questions about R programming based on the R-help mailing list. "
        "Do not ask the user for more information, but retrieve emails from the R-help mailing list archives. "
        f"You can retrieve emails from {start} to {end} using a search query. "
        "Write a search query based on the user's question, but do not answer the question just yet. "
        "For questions about differences or comparison between X and Y, retrieve emails about X and Y. "
        "Only use a year range that can be inferred from the user's question. "
        "Use the search_query argument to specify months. "
        # This confuses gpt-4o-mini (empty search_query - token problem?)
        # "Use 3-letter month abbreviations (Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec). "
        "Example: If the current month is July 2025, then the previous month is June 2025. "
        "You can retrieve emails from June 2025 using retrieve_emails(search_query='June', start_year=2025, end_year=2025). "
        "If you decide not to retrieve emails, tell the user why and suggest how to improve their question to chat with the R-help mailing list. "
    )
    # A sanity check that we don't have unassigned variables
    # (this causes KeyError in parsing by ToolCallingLLM)
    matches = re.findall(r"\{.*?\}", "".join(retrieve_prompt))
    if matches:
        raise ValueError(f"Unassigned variables in prompt: {' '.join(matches)}")
    return retrieve_prompt


def answer_prompt():
    """Return system prompt for generate step"""
    answer_prompt = (
        f"The current date is {date.today()}. "
        "You are a helpful RAG chatbot designed to answer questions about R programming based on the R-help mailing list. "
        "Summarize the retrieved emails from the R-help mailing list archives to answer the user's question or query. "
        "Example: If retrieved emails are from Jan 2024, Dec 2024, and Jan 2025, use only emails from Jan 2025 to answer questions about January 2025. "
        "Tell the user if there are no retrieved emails or if you are unable to answer the question based on the information in the emails. "
        "Do not give an answer based on your own knowledge or memory, and do not include examples that aren't based on the retrieved emails. "
        "Example: For a question about writing formulas for lm(), make your answer about formulas for lm() from the retrieved emails. "
        "Do not respond with packages that are only listed under sessionInfo, session info, or other attached packages. "
        "Include citations (email senders and dates) in your response. "
        "Respond with 300 words maximum and 30 lines of code maximum and include any relevant URLs from the retrieved emails. "
        "Use answer_with_citations to respond with the answer and citations of emails used in your response. "
    )
    matches = re.findall(r"\{.*?\}", "".join(answer_prompt))
    if matches:
        raise ValueError(f"Unassigned variables in prompt: {' '.join(matches)}")
    return answer_prompt


# Prompt template for SmolLM3 with tools

smollm3_tools_template = """

    ### Tools

    You may call one or more functions to assist with the user query.

    You have access to the following tools:

    {tools}

    You must always select one of the above tools and respond with only a JSON object matching the following schema:

    {{
      "tool": <name of the selected tool>,
      "tool_input": <parameters for the selected tool, matching the tool's JSON schema>
    }}
    """
