from typing import Optional

import click
import llm
from anthropic import Anthropic
from pydantic import Field, field_validator

@llm.hookimpl
def register_models(register):
    # Registering models as per the latest naming conventions
    register(Claude("claude-instant-1.2"), aliases=("claude-instant",))
    register(Claude("claude-2.1"), aliases=("claude",))

class Claude(llm.Model):
    needs_key = "claude"
    key_env_var = "ANTHROPIC_API_KEY"
    can_stream = True

    class Options(llm.Options):
        max_tokens: Optional[int] = Field(
            description="The maximum number of tokens for the model to generate",
            default=4096,  # Adjusted to the maximum allowed for claude-2.1
        )

        @field_validator("max_tokens")
        def validate_max_tokens(cls, max_tokens):
            if not (0 < max_tokens <= 4096):  # Updated maximum limit
                raise ValueError("max_tokens must be in range 1-4096 for claude-2.1")
            return max_tokens

    def __init__(self, model_id):
        self.model_id = model_id

    def generate_prompt_messages(self, prompt, conversation):
        # Generate a list of message dictionaries based on conversation history
        messages = []
        current_system = None
        if conversation is not None:
            for prev_response in conversation.responses:
                if (prev_response.prompt.system and prev_response.prompt.system != current_system):
                    current_system = prev_response.prompt.system
                messages.append({"role": "user", "content": prev_response.prompt.prompt})
                messages.append({"role": "assistant", "content": prev_response.text()})
        if prompt.system and prompt.system != current_system:
            current_system = prompt.system
        messages.append({"role": "user", "content": prompt.prompt})
        
        return messages, current_system

    def execute(self, prompt, stream, response, conversation):
        anthropic = Anthropic(api_key=self.get_key())

        messages, system_prompt = self.generate_prompt_messages(prompt, conversation)

        if stream:
            # Handling streaming responses
            with anthropic.beta.messages.stream(
                max_tokens=prompt.options.max_tokens,
                messages=messages,
                system=system_prompt,
                model=self.model_id
            ) as stream_response:
                for text in stream_response.text_stream:
                    yield text
        else:
            # Handling non-streaming response
            message_response = anthropic.beta.messages.create(
                model=self.model_id,
                max_tokens=prompt.options.max_tokens,
                messages=messages,
                system=system_prompt
            )
            # Concatenating text from content blocks
            yield "".join(content_block['text'] for content_block in message_response.content)

    def __str__(self):
        return "Anthropic: {}".format(self.model_id)
