from datetime import date
from util import get_sources, get_start_end_months
import re


def check_prompt(prompt, chat_model, think):
    """Check for unassigned variables and add /no_think if needed"""
    # A sanity check that we don't have unassigned variables
    # (this causes KeyError in parsing by ToolCallingLLM)
    matches = re.findall(r"\{.*?\}", " ".join(prompt))
    if matches:
        raise ValueError(f"Unassigned variables in prompt: {' '.join(matches)}")
    # Check if we should add /no_think to turn off thinking mode
    if hasattr(chat_model, "model_id"):
        model_id = chat_model.model_id
        if ("SmolLM" in model_id or "Qwen" in model_id) and not think:
            prompt = "/no_think\n" + prompt
    return prompt


def query_prompt(chat_model, think=False):
    """Return system prompt for query step"""

    # Get start and end months from database
    start, end = get_start_end_months(get_sources())

    prompt = (
        f"Today Date: {date.today()}. "
        "You are a helpful assistant designed to get information about R programming from the R-help mailing list archives. "
        "Write a search query to retrieve emails relevant to the user's question. "
        "Do not answer the user's question and do not ask the user for more information. "
        # gpt-4o-mini thinks last two months aren't available with this: "Emails from from {start} to {end} are available for retrieval. "
        f"The emails available for retrieval are from {start} to {end}. "
        "For questions about differences or comparison between X and Y, retrieve emails about X and Y using separate tool calls. "
        "For general summaries, use retrieve_emails(search_query='R'). "
        "For specific questions, use retrieve_emails(search_query=<specific topic>). "
        "For questions about years, use retrieve_emails(search_query=, start_year=, end_year=) (this month is this year). "
        "For questions about months, use 3-letter abbreviations (Jan...Dec) for the 'month' argument. "
        "Even if retrieved emails are available, you should retrieve more emails to answer the most recent question. "  # Qwen
        # "You must perform the search yourself. Do not tell the user how to retrieve emails. "  # Qwen
        "Do not use your memory or knowledge to answer the user's question. Only retrieve emails based on the user's question. "  # Qwen
        # "If you decide not to retrieve emails, tell the user why and suggest how to improve their question to chat with the R-help mailing list. "
    )
    prompt = check_prompt(prompt, chat_model, think)

    return prompt


def answer_prompt(chat_model, think=False, with_tools=False):
    """Return system prompt for answer step"""
    prompt = (
        f"Today Date: {date.today()}. "
        "You are a helpful chatbot designed to answer questions about R programming based on the R-help mailing list archives. "
        "Summarize the retrieved emails to answer the user's question or query. "
        "If any of the retrieved emails are irrelevant (e.g. wrong dates), then do not use them. "
        "Tell the user if there are no retrieved emails or if you are unable to answer the question based on the information in the emails. "
        "Do not give an answer based on your own knowledge or memory, and do not include examples that aren't based on the retrieved emails. "
        "Example: For a question about using lm(), take examples of lm() from the retrieved emails to answer the user's question. "
        # "Do not respond with packages that are only listed under sessionInfo, session info, or other attached packages. "
        "Summarize the content of the emails rather than copying the headers. "  # Qwen
        "You must include inline citations (email senders and dates) in each part of your response. "
        "Only answer general questions about R if the answer is in the retrieved emails. "
        "Your response can include URLs, but make sure they are unchanged from the retrieved emails. "  # Qwen
        "Respond with 500 words maximum and 50 lines of code maximum. "
    )
    if with_tools:
        prompt = (
            f"{prompt}"
            "Use answer_with_citations to provide the complete answer and all citations used. "
        )
    prompt = check_prompt(prompt, chat_model, think)

    return prompt


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
}},
{{
    "tool": <function-name>,
    "tool_input": <args-json-object>
}}

"""

# Prompt template for Gemma/Qwen with tools
# Based on https://ai.google.dev/gemma/docs/capabilities/function-calling
generic_tools_template = """

### Functions

You have access to functions. If you decide to invoke any of the function(s), you MUST put it in the format of

{{
    "tool": <function-name>,
    "tool_input": <args-json-object>
}},
{{
    "tool": <function-name>,
    "tool_input": <args-json-object>
}}

You SHOULD NOT include any other text in the response if you call a function

{tools}
"""
