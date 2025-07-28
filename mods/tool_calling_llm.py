import re
import json
import uuid
import warnings
from abc import ABC
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
)

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import (
    SystemMessage,
    AIMessage,
    BaseMessage,
    BaseMessageChunk,
    ToolCall,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.prompts import SystemMessagePromptTemplate
from pydantic import BaseModel
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool

DEFAULT_SYSTEM_TEMPLATE = """You have access to the following tools:

{tools}

You must always select one of the above tools and respond with only a JSON object matching the following schema:

{{
  "tool": <name of the selected tool>,
  "tool_input": <parameters for the selected tool, matching the tool's JSON schema>
}}
"""  # noqa: E501


def _is_pydantic_class(obj: Any) -> bool:
    """
    Checks if the tool provided is a Pydantic class.
    """
    return isinstance(obj, type) and (
        issubclass(obj, BaseModel) or BaseModel in obj.__bases__
    )


def _is_pydantic_object(obj: Any) -> bool:
    """
    Checks if the tool provided is a Pydantic object.
    """
    return isinstance(obj, BaseModel)


def RawJSONDecoder(index):
    class _RawJSONDecoder(json.JSONDecoder):
        end = None

        def decode(self, s, *_):
            data, self.__class__.end = self.raw_decode(s, index)
            return data

    return _RawJSONDecoder


def extract_json(s, index=0):
    while (index := s.find("{", index)) != -1:
        try:
            yield json.loads(s, cls=(decoder := RawJSONDecoder(index)))
            index = decoder.end
        except json.JSONDecodeError:
            index += 1


def parse_json_garbage(s: str) -> Any:
    # Find the first occurrence of a JSON opening brace or bracket
    candidates = list(extract_json(s))
    if len(candidates) >= 1:
        return candidates[0]

    raise ValueError("Not a valid JSON string")


def extract_think(content):
    # Added by Cursor 20250726 jmd
    # Extract content within <think>...</think>
    think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
    think_text = think_match.group(1).strip() if think_match else ""
    # Extract text after </think>
    if think_match:
        post_think = content[think_match.end() :].lstrip()
    else:
        # Check if content starts with <think> but missing closing tag
        if content.strip().startswith("<think>"):
            # Extract everything after <think>
            think_start = content.find("<think>") + len("<think>")
            think_text = content[think_start:].strip()
            post_think = ""
        else:
            # No <think> found, so return entire content as post_think
            post_think = content
    return think_text, post_think


class ToolCallingLLM(BaseChatModel, ABC):
    """ToolCallingLLM mixin to enable tool calling features on non tool calling models.

    Note: This is an incomplete mixin and should not be used directly. It must be used to extent an existing Chat Model.

    Setup:
      Install dependencies for your Chat Model.
      Any API Keys or setup needed for your Chat Model is still applicable.

    Key init args — completion params:
      Refer to the documentation of the Chat Model you wish to extend with Tool Calling.

    Key init args — client params:
      Refer to the documentation of the Chat Model you wish to extend with Tool Calling.

    See full list of supported init args and their descriptions in the params section.

    Instantiate:
      ```
      # Example implementation using LiteLLM
      from langchain_community.chat_models import ChatLiteLLM

      class LiteLLMFunctions(ToolCallingLLM, ChatLiteLLM):

          def __init__(self, **kwargs: Any) -> None:
              super().__init__(**kwargs)

          @property
          def _llm_type(self) -> str:
              return "litellm_functions"

      llm = LiteLLMFunctions(model="ollama/phi3")
      ```

    Invoke:
      ```
      messages = [
        ("human", "What is the capital of France?")
      ]
      llm.invoke(messages)
      ```
      ```
      AIMessage(content='The capital of France is Paris.', id='run-497d0e1a-d63b-45e8-9c8b-5e76d99b9468-0')
      ```

    Tool calling:
      ```
      from langchain_core.pydantic_v1 import BaseModel, Field

      class GetWeather(BaseModel):
          '''Get the current weather in a given location'''

          location: str = Field(..., description="The city and state, e.g. San Francisco, CA")

      class GetPopulation(BaseModel):
          '''Get the current population in a given location'''

          location: str = Field(..., description="The city and state, e.g. San Francisco, CA")

      llm_with_tools = llm.bind_tools([GetWeather, GetPopulation])
      ai_msg = llm_with_tools.invoke("Which city is hotter today and which is bigger: LA or NY?")
      ai_msg.tool_calls
      ```
      ```
      [{'name': 'GetWeather', 'args': {'location': 'Austin, TX'}, 'id': 'call_25ed526917b94d8fa5db3fe30a8cf3c0'}]
      ```

    Response metadata
      Refer to the documentation of the Chat Model you wish to extend with Tool Calling.

    """  # noqa: E501

    tool_system_prompt_template: str = DEFAULT_SYSTEM_TEMPLATE
    # Suffix to add to the system prompt that is not templated 20250717 jmd
    system_message_suffix: str = ""

    override_bind_tools: bool = True

    def __init__(self, **kwargs: Any) -> None:
        override_bind_tools = True
        if "override_bind_tools" in kwargs:
            override_bind_tools = kwargs["override_bind_tools"]
            del kwargs["override_bind_tools"]
        super().__init__(**kwargs)
        self.override_bind_tools = override_bind_tools

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        if self.override_bind_tools:
            return self.bind(functions=tools, **kwargs)
        else:
            return super().bind_tools(tools, **kwargs)

    def _generate_system_message_and_functions(
        self,
        kwargs: Dict[str, Any],
    ) -> Tuple[BaseMessage, List]:
        functions = kwargs.get("tools", kwargs.get("functions", []))
        functions = [
            (
                fn["function"]
                if (
                    not _is_pydantic_class(fn)
                    and not _is_pydantic_object(fn)
                    and "name" not in fn.keys()
                    and "function" in fn.keys()
                    and "name" in fn["function"].keys()
                )
                else fn
            )
            for fn in functions
        ]

        # langchain_openai/chat_models/base.py:
        # NOTE: Using bind_tools is recommended instead, as the `functions` and
        # `function_call` request parameters are officially marked as
        # deprecated by OpenAI.

        # if "functions" in kwargs:
        #    del kwargs["functions"]
        # if "function_call" in kwargs:
        #    functions = [
        #        fn for fn in functions if fn["name"] == kwargs["function_call"]["name"]
        #    ]
        #    if not functions:
        #        raise ValueError(
        #            "If `function_call` is specified, you must also pass a "
        #            "matching function in `functions`."
        #        )
        #    del kwargs["function_call"]

        functions = [convert_to_openai_tool(fn) for fn in functions]
        system_message_prompt_template = SystemMessagePromptTemplate.from_template(
            self.tool_system_prompt_template
        )
        system_message = system_message_prompt_template.format(
            tools=json.dumps(functions, indent=2)
        )
        # Add extra context after the formatted system message 20250717 jmd
        system_message = SystemMessage(
            system_message.content + self.system_message_suffix
        )
        return system_message, functions

    def _process_response(
        self, response_message: BaseMessage, functions: List[Dict]
    ) -> AIMessage:
        if not isinstance(response_message.content, str):
            raise ValueError("ToolCallingLLM does not support non-string output.")

        # Extract <think>...</think> content and text after </think> for further processing 20250726 jmd
        think_text, post_think = extract_think(response_message.content)

        # Parse output for JSON
        try:
            parsed_json_result = json.loads(post_think)
        except json.JSONDecodeError:
            try:
                print("parse_json_garbage for content:")
                print(post_think)
                parsed_json_result = parse_json_garbage(post_think)
            except Exception:
                # Return entire response if JSON is missing or wasn't parsed
                return AIMessage(content=response_message.content)

        print("parsed_json_result")
        print(parsed_json_result)

        # Get tool name from output
        called_tool_name = (
            parsed_json_result["tool"]
            if "tool" in parsed_json_result
            else parsed_json_result["name"] if "name" in parsed_json_result else None
        )

        # Check if tool name is in functions list
        called_tool = next(
            (fn for fn in functions if fn["function"]["name"] == called_tool_name), None
        )
        if called_tool is None:
            # Issue a warning and return the generated content 20250727 jmd
            warnings.warn(
                f"Tool {called_tool} called from {self.model} output not in functions list"
            )
            return AIMessage(content=response_message.content)

        # Get tool arguments from output
        called_tool_arguments = (
            parsed_json_result["tool_input"]
            if "tool_input" in parsed_json_result
            else (
                parsed_json_result["parameters"]
                if "parameters" in parsed_json_result
                else {}
            )
        )

        # Put together response message
        response_message_with_functions = AIMessage(
            content=f"<think>\n{think_text}\n</think>",
            tool_calls=[
                ToolCall(
                    name=called_tool_name,
                    args=called_tool_arguments,
                    id=f"call_{str(uuid.uuid4()).replace('-', '')}",
                )
            ],
        )

        return response_message_with_functions

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        system_message, functions = self._generate_system_message_and_functions(kwargs)
        response_message = super()._generate(  # type: ignore[safe-super]
            [system_message] + messages, stop=stop, run_manager=run_manager, **kwargs
        )
        response = self._process_response(
            response_message.generations[0].message, functions
        )
        return ChatResult(generations=[ChatGeneration(message=response)])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        system_message, functions = self._generate_system_message_and_functions(kwargs)
        response_message = await super()._agenerate(
            [system_message] + messages, stop=stop, run_manager=run_manager, **kwargs
        )
        response = self._process_response(
            response_message.generations[0].message, functions
        )
        return ChatResult(generations=[ChatGeneration(message=response)])

    async def astream(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[BaseMessageChunk]:
        system_message, functions = self._generate_system_message_and_functions(kwargs)
        generation: Optional[BaseMessageChunk] = None
        async for chunk in super().astream(
            [system_message] + super()._convert_input(input).to_messages(),
            stop=stop,
            **kwargs,
        ):
            if generation is None:
                generation = chunk
            else:
                generation += chunk
        assert generation is not None
        response = self._process_response(generation, functions)
        yield cast(BaseMessageChunk, response)
