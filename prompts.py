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
        f"Today Date: {date.today()}."
        "You are a helpful RAG chatbot designed to answer questions about R programming based on the R-help mailing list."
        "Do not ask the user for more information, but retrieve emails from the R-help mailing list archives."
        # gpt-4o-mini says last two months aren't available with this: Emails from from {start} to {end} are available for retrieval.
        f"The emails available for retrieval are from {start} to {end}."
        "Write a search query based on the user's question, but do not answer the question just yet."
        "For questions about differences or comparison between X and Y, retrieve emails about X and Y."
        "For general summaries, use retrieve_emails(search_query='R')."
        "For specific questions, use retrieve_emails(search_query=<specific topic>)."
        "For questions about years, use retrieve_emails(search_query=, start_year=, end_year=) (this month is this year)."
        "For questions about months, use 3-letter abbreviations (Jan..Dec) for the 'month' argument."
        "If you decide not to retrieve emails, tell the user why and suggest how to improve their question to chat with the R-help mailing list."
    )
    # A sanity check that we don't have unassigned variables
    # (this causes KeyError in parsing by ToolCallingLLM)
    matches = re.findall(r"\{.*?\}", " ".join(retrieve_prompt))
    if matches:
        raise ValueError(f"Unassigned variables in prompt: {' '.join(matches)}")
    return retrieve_prompt


def answer_prompt(with_tools=True):
    """Return system prompt for generate step"""
    answer_prompt = (
        f"Today Date: {date.today()}."
        "You are a helpful RAG chatbot designed to answer questions about R programming based on the R-help mailing list."
        "Summarize the retrieved emails from the R-help mailing list archives to answer the user's question or query."
        "If any of the retrieved emails are irrelevant (e.g. wrong dates), then do not use them."
        "Tell the user if there are no retrieved emails or if you are unable to answer the question based on the information in the emails."
        "Do not give an answer based on your own knowledge or memory, and do not include examples that aren't based on the retrieved emails."
        "Example: For a question about writing formulas for lm(), make your answer about formulas for lm() from the retrieved emails."
        "Do not respond with packages that are only listed under sessionInfo, session info, or other attached packages."
        "Include inline citations (email senders and dates) in your response."
        "Only answer general questions about R if the answer is given in the retrieved emails."
        "Respond with 300 words maximum and 30 lines of code maximum and include any relevant URLs from the retrieved emails."
    )
    if with_tools:
        answer_prompt += "Use answer_with_citations to provide the complete answer and all citations used. "
    matches = re.findall(r"\{.*?\}", " ".join(answer_prompt))
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
