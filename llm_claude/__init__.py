import click
import httpx
import llm

from .vendored_anthropic import AI_PROMPT, HUMAN_PROMPT, Anthropic


@llm.hookimpl
def register_models(register):
    # https://docs.anthropic.com/claude/reference/selecting-a-model
    # Family           Latest major version    Latest full version
    # Claude Instant   claude-instant-1        claude-instant-1.1
    # Claude           claude-2                claude-2.0
    register(Claude("claude-instant-1"), aliases=("claude-instant",))
    register(Claude("claude-2"), aliases=("claude",))


class Claude(llm.Model):
    needs_key = "claude"
    key_env_var = "ANTHROPIC_API_KEY"
    can_stream = True

    def __init__(self, model_id):
        self.model_id = model_id

    def generate_prompt_messages(self, prompt, conversation):
        if conversation:
            for response in conversation.responses:
                yield self.build_prompt(response.prompt.prompt, response.text())

        yield self.build_prompt(prompt)

    def build_prompt(self, human, ai=""):
        return f"{HUMAN_PROMPT} {human}{AI_PROMPT}{ai}"

    def execute(self, prompt, stream, response, conversation):
        anthropic = Anthropic(api_key=self.get_key())

        prompt_str = "".join(self.generate_prompt_messages(prompt.prompt, conversation))

        completion = anthropic.completions.create(
            model=self.model_id,
            max_tokens_to_sample=300,
            prompt=prompt_str,
            stream=stream,
        )
        if stream:
            for comp in completion:
                yield comp.completion
        else:
            yield completion.completion

    def __str__(self):
        return "Anthropic: {}".format(self.model_id)
