from typing import Optional
import llm
from anthropic import Anthropic
from pydantic import Field, field_validator


@llm.hookimpl
def register_models(register):
    # https://docs.anthropic.com/claude/docs/models-overview
    register(Claude("claude-instant-1.2"), aliases=("claude-instant",))
    register(Claude("claude-2.0"))
    register(Claude("claude-2.1"), aliases=("claude-2",))
    register(Claude("claude-3-opus-20240229"), aliases=("claude", "claude-3", "opus", "claude-opus"))
    register(Claude("claude-3-sonnet-20240229"), aliases=("sonnet", "claude-sonnet"))
    # TODO haiku when it's released


class _ClaudeOptions(llm.Options):
    max_tokens: Optional[int] = Field(
        description="The maximum number of tokens to generate before stopping",
        default=4_096,
    )

    temperature: Optional[float] = Field(
        description="Amount of randomness injected into the response. Defaults to 1.0. Ranges from 0.0 to 1.0. Use temperature closer to 0.0 for analytical / multiple choice, and closer to 1.0 for creative and generative tasks. Note that even with temperature of 0.0, the results will not be fully deterministic.",
        default=1.0,
    )

    top_p: Optional[float] = Field(
        description="Use nucleus sampling. In nucleus sampling, we compute the cumulative distribution over all the options for each subsequent token in decreasing probability order and cut it off once it reaches a particular probability specified by top_p. You should either alter temperature or top_p, but not both. Recommended for advanced use cases only. You usually only need to use temperature.",
        default=None,
    )

    top_k: Optional[int] = Field(
        description="Only sample from the top K options for each subsequent token. Used to remove 'long tail' low probability responses. Recommended for advanced use cases only. You usually only need to use temperature.",
        default=None,
    )

    @field_validator("max_tokens")
    def validate_max_tokens(cls, max_tokens):
        if not (0 < max_tokens <= 4_096):
            raise ValueError("max_tokens must be in range 1-4,096")
        return max_tokens

    @field_validator("temperature")
    def validate_temperature(cls, temperature):
        if not (0.0 <= temperature <= 1.0):
            raise ValueError("temperature must be in range 0.0-1.0")
        return temperature

    @field_validator("top_p")
    def validate_top_p(cls, top_p):
        if top_p is not None and not (0.0 <= top_p <= 1.0):
            raise ValueError("top_p must be in range 0.0-1.0")
        return top_p

    @field_validator("top_k")
    def validate_top_k(cls, top_k):
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        return top_k


class Claude(llm.Model):
    needs_key = "claude"
    key_env_var = "ANTHROPIC_API_KEY"
    can_stream = True

    class Options(_ClaudeOptions):
        ...

    def __init__(self, model_id):
        self.model_id = model_id

    def generate_prompt_messages(self, prompt, conversation):
        messages = []
        if conversation:
            for response in conversation.responses:
                messages.append({"role": "user", "content": response.prompt.prompt})
                messages.append({"role": "assistant", "content": response.text()})
        messages.append({"role": "user", "content": prompt})
        return messages

    def execute(self, prompt, stream, response, conversation):
        anthropic = Anthropic(api_key=self.get_key())
        messages = self.generate_prompt_messages(prompt.prompt, conversation)
        completion = anthropic.messages.create(
            model=self.model_id,
            max_tokens=prompt.options.max_tokens,
            messages=messages,
            stream=stream,
        )
        if stream:
            for comp in completion:
                if hasattr(comp, "content_block"):
                    response = comp.content_block.text
                    yield response
                elif hasattr(comp, "delta"):
                    if hasattr(comp.delta, "text"):
                        yield comp.delta.text
        else:
            yield completion.completion

    def __str__(self):
        return "Anthropic: {}".format(self.model_id)
