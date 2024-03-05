import os
from typing import List, Tuple
from unittest.mock import MagicMock, Mock, patch

import pytest
from click.testing import CliRunner
from llm import Prompt, Response, get_model
from llm.cli import cli

from llm_claude import Claude


@patch("llm_claude.Anthropic")
def test_claude_response(mock_anthropic):
    mock_response = MagicMock()
    mock_response.completion = "hello"
    mock_anthropic.return_value.messages.create.return_value.__iter__.return_value = [
        mock_response
    ]
    prompt = Prompt("hello", "", options=Claude.Options())
    model = Claude("claude-2")
    model.key = "key"
    model_response = model.response(prompt)
    # breakpoint()
    items = list(model_response)

    mock_anthropic.return_value.messages.create.assert_called_with(
        model="claude-2",
        max_tokens=10_000,
        messages=[{"role": "user", "content": "hello"}],
        stream=True,
    )
    # breakpoint()

    assert items == ["hello"]


@pytest.mark.parametrize("max_tokens", (1, 500_000, 1_000_000))
@patch("llm_claude.Anthropic")
def test_with_max_tokens(mock_anthropic, max_tokens):
    mock_response = MagicMock()
    mock_response.completion = "hello"
    mock_anthropic.return_value.messages.create.return_value.__iter__.return_value = [
        mock_response
    ]
    prompt = Prompt(
        "hello", "", options=Claude.Options(max_tokens=max_tokens)
    )
    model = Claude("claude-2")
    model.key = "key"
    model_response = model.response(prompt)
    items = list(model_response)

    mock_anthropic.return_value.messages.create.assert_called_with(
        model="claude-2",
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": "hello"}],
        stream=True,
    )

    assert items == ["hello"]


@pytest.mark.parametrize("max_tokens", (0, 1_000_001))
@patch("llm_claude.Anthropic")
def test_invalid_max_tokens(mock_anthropic, max_tokens):
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "two dog emoji",
            "-m",
            "claude",
            "-o",
            "max_tokens",
            max_tokens,
        ],
    )
    assert result.exit_code == 1
    assert (
        result.output
        == "Error: max_tokens\n  Value error, max_tokens must be in range 1-1,000,000\n"
    )


@patch.dict(os.environ, {"ANTHROPIC_API_KEY": "X"})
@patch("llm_claude.Anthropic")
def test_claude_prompt(mock_anthropic):
    mock_response = Mock()
    mock_response.completion = "🐶🐶"
    mock_anthropic.return_value.messages.create.return_value = mock_response
    runner = CliRunner()
    result = runner.invoke(cli, ["two dog emoji", "-m", "claude", "--no-stream"])
    assert result.exit_code == 0, result.output
    assert result.output == "🐶🐶\n"


@pytest.mark.parametrize(
    "prompt, conversation_messages, expected",
    (
        ("hello", [], [{"role": "user", "content": "hello"}]),
        (
            "hello 2",
            [("user 1", "response 1")],
            [
                {"role": "user", "content": "user 1"},
                {"role": "assistant", "content": "response 1"},
                {"role": "user", "content": "hello 2"},
            ],
        ),
        (
            "hello 3",
            [("user 1", "response 1"), ("user 2", "response 2")],
            [
                {"role": "user", "content": "user 1"},
                {"role": "assistant", "content": "response 1"},
                {"role": "user", "content": "user 2"},
                {"role": "assistant", "content": "response 2"},
                {"role": "user", "content": "hello 3"},
            ],
        ),
    ),
)
def test_generate_prompt_messages(
    prompt: str, conversation_messages: List[Tuple[str, str]], expected: List[dict]
):
    model = get_model("claude")
    conversation = None
    if conversation_messages:
        conversation = model.conversation()
        for prev_prompt, prev_response in conversation_messages:
            conversation.responses.append(
                Response.fake(
                    prompt=prev_prompt,
                    model=model,
                    system=None,
                    response=prev_response,
                )
            )
    messages = model.generate_prompt_messages(prompt, conversation)
    assert list(messages) == expected
