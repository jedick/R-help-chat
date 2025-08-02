from transformers.pipelines.text_generation import Chat
from transformers import TextGenerationPipeline
from typing import Dict


class MyTextGenerationPipeline(TextGenerationPipeline):
    """
    This subclass overrides the preprocess method to add pad_to_multiple_of=8 to tokenizer_kwargs.
    Fix for: "RuntimeError: p.attn_bias_ptr is not correctly aligned"
    https://github.com/google-deepmind/gemma/issues/169
    """

    def preprocess(
        self,
        prompt_text,
        prefix="",
        handle_long_generation=None,
        add_special_tokens=None,
        truncation=None,
        padding=None,
        max_length=None,
        continue_final_message=None,
        **generate_kwargs,
    ):
        # Only set non-None tokenizer kwargs, so as to rely on the tokenizer's defaults
        tokenizer_kwargs = {
            "add_special_tokens": add_special_tokens,
            "truncation": truncation,
            "padding": padding,
            "max_length": max_length,
            "pad_to_multiple_of": 8,
        }
        tokenizer_kwargs = {
            key: value for key, value in tokenizer_kwargs.items() if value is not None
        }

        if isinstance(prompt_text, Chat):
            tokenizer_kwargs.pop(
                "add_special_tokens", None
            )  # ignore add_special_tokens on chats
            # If the user passes a chat that ends in an assistant message, we treat it as a prefill by default
            # because very few models support multiple separate, consecutive assistant messages
            if continue_final_message is None:
                continue_final_message = prompt_text.messages[-1]["role"] == "assistant"
            inputs = self.tokenizer.apply_chat_template(
                prompt_text.messages,
                add_generation_prompt=not continue_final_message,
                continue_final_message=continue_final_message,
                return_dict=True,
                return_tensors=self.framework,
                **tokenizer_kwargs,
            )
        else:
            inputs = self.tokenizer(
                prefix + prompt_text, return_tensors=self.framework, **tokenizer_kwargs
            )

        inputs["prompt_text"] = prompt_text

        if handle_long_generation == "hole":
            cur_len = inputs["input_ids"].shape[-1]
            if "max_new_tokens" in generate_kwargs:
                new_tokens = generate_kwargs["max_new_tokens"]
            else:
                new_tokens = (
                    generate_kwargs.get("max_length", self.generation_config.max_length)
                    - cur_len
                )
                if new_tokens < 0:
                    raise ValueError("We cannot infer how many new tokens are expected")
            if cur_len + new_tokens > self.tokenizer.model_max_length:
                keep_length = self.tokenizer.model_max_length - new_tokens
                if keep_length <= 0:
                    raise ValueError(
                        "We cannot use `hole` to handle this generation the number of desired tokens exceeds the"
                        " models max length"
                    )

                inputs["input_ids"] = inputs["input_ids"][:, -keep_length:]
                if "attention_mask" in inputs:
                    inputs["attention_mask"] = inputs["attention_mask"][
                        :, -keep_length:
                    ]

        return inputs
