from datetime import date
from util import get_sources, get_start_end_months
import re


def retrieve_prompt(compute_mode):
    """Return system prompt for query step

    Args:
        compute_mode: Compute mode for embedding model (cloud or edge)
    """

    # Get start and end months from database
    start, end = get_start_end_months(get_sources())

    retrieve_prompt = (
        f"Today Date: {date.today()}. "
        "You are a helpful RAG chatbot designed to answer questions about R programming based on the R-help mailing list. "
        "Do not ask the user for more information, but retrieve emails from the R-help mailing list archives. "
        f"The emails available for retrieval are from {start} to {end}. "
        "Write a search query based on the user'r question, but do not answer the question just yet. "
        "For questions about differences or comparison between X and Y, retrieve emails about X and Y. "
        "Use the 'months' argument to search for months. "
        # This confuses gpt-4o-mini (empty search_query - token problem?)
        # "Use 3-letter month abbreviations (Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec). "
        "Example: retrieve emails about R in <month> using retrieve_emails(search_query='R', months=<month>). "
        "Try to infer years from the user's question (last month or this month is this year). "
        "If you can infer years, use retrieve_emails(search_query=, start_year=, end_year=), otherwise retrieve_emails(search_query=). "
        "If you decide not to retrieve emails, tell the user why and suggest how to improve their question to chat with the R-help mailing list. "
    )
    # A sanity check that we don't have unassigned variables
    # (this causes KeyError in parsing by ToolCallingLLM)
    matches = re.findall(r"\{.*?\}", "".join(retrieve_prompt))
    if matches:
        raise ValueError(f"Unassigned variables in prompt: {' '.join(matches)}")
    return retrieve_prompt


def answer_prompt(with_tools=True):
    """Return system prompt for generate step"""
    answer_prompt = (
        f"Today Date: {date.today()}. "
        "You are a helpful RAG chatbot designed to answer questions about R programming based on the R-help mailing list. "
        "Summarize the retrieved emails from the R-help mailing list archives to answer the user's question or query. "
        "If any of the retrieved emails are irrelevant (e.g. wrong dates), then do not use them. "
        "Tell the user if there are no retrieved emails or if you are unable to answer the question based on the information in the emails. "
        "Do not give an answer based on your own knowledge or memory, and do not include examples that aren't based on the retrieved emails. "
        "Example: For a question about writing formulas for lm(), make your answer about formulas for lm() from the retrieved emails. "
        "Do not respond with packages that are only listed under sessionInfo, session info, or other attached packages. "
        "Include inline citations (email senders and dates) in your response. "
        "Only answer general questions about R if the answer is given in the retrieved emails. "
        "Respond with 300 words maximum and 30 lines of code maximum and include any relevant URLs from the retrieved emails. "
    )
    if with_tools:
        answer_prompt += (
            "Use answer_with_citations to provide the answer and all citations used. "
        )
    matches = re.findall(r"\{.*?\}", "".join(answer_prompt))
    if matches:
        raise ValueError(f"Unassigned variables in prompt: {' '.join(matches)}")
    return answer_prompt


# Prompt template for SmolLM3 with tools
# The first two lines, <function-name>, and <args-json-object> are from the apply_chat_template for HuggingFaceTB/SmolLM3-3B
# The other lines (You have, {tools}, You must), "tool", and "tool_input" are from tool_calling_llm.py
smollm3_tools_template = """

### Tools

You may call one or more functions to assist with the user query.

You have access to the following tools:

{tools}

You must always select one of the above tools and respond with only a JSON object matching the following schema:

{{
    "tool": <function-name>,
    "tool_input": <args-json-object>
}}

"""

# Prompt template for Gemma-3 with tools
# Based on https://ai.google.dev/gemma/docs/capabilities/function-calling
gemma_tools_template = """

### Functions

You have access to functions. If you decide to invoke any of the function(s), you MUST put it in the format of

{{
    "tool": <function-name>,
    "tool_input": <args-json-object>
}}

You SHOULD NOT include any other text in the response if you call a function

{tools}
"""
