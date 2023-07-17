import llm
import httpx


@llm.hookimpl
def register_models(register):
    register(Claude())


class Claude(llm.Model):
    needs_key = "claude"
    key_env_var = "ANTHROPIC_API_KEY"

    model_id = "claude"

    def execute(self, prompt, stream, response, conversation):
        url = "https://api.anthropic.com/v1/complete"
        headers = {
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
            "x-api-key": self.get_key(),
        }

        data = {
            "model": "claude-2",
            "prompt": "\n\nHuman: {prompt}\n\nAssistant:",
            "max_tokens_to_sample": 256,
            "stream": False,
        }
        response = httpx.post(url, headers=headers, json=data)

        return [response.json()['completion']]
