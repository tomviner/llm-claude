# llm-claude

# Deprecated in favour of llm-claude-3

Please see https://github.com/simonw/llm-claude-3/ instead

---

[![PyPI](https://img.shields.io/pypi/v/llm-claude.svg)](https://pypi.org/project/llm-claude/)
[![Changelog](https://img.shields.io/github/v/release/tomviner/llm-claude?include_prereleases&label=changelog)](https://github.com/tomviner/llm-claude/releases)
[![Tests](https://github.com/tomviner/llm-claude/workflows/Test/badge.svg)](https://github.com/tomviner/llm-claude/actions?query=workflow%3ATest)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/tomviner/llm-claude/blob/main/LICENSE)

Plugin for [LLM](https://llm.datasette.io/) adding support for Anthropic's Claude models.

## Installation

Install this plugin in the same environment as LLM.
```bash
llm install llm-claude
```
## Configuration

You will need an API key from Anthropic. Request access at https://www.anthropic.com/earlyaccess then go to https://console.anthropic.com/account/keys

You can set that as an environment variable called `ANTHROPIC_API_KEY`, or add it to the `llm` set of saved keys using:

```bash
llm keys set claude
```
```
Enter key: <paste key here>
```

## Usage

This plugin adds models called `claude` and `claude-instant`.

Anthropic [describes them as](https://docs.anthropic.com/claude/reference/selecting-a-model):

> two families of models, both of which support 100,000 token context windows:
> - **Claude Instant**: low-latency, high throughput
> - **Claude**: superior performance on tasks that require complex reasoning

You can query them like this:

```bash
llm -m claude-instant "Ten great names for a new space station"
```

```bash
llm -m claude "Compare and contrast the leadership styles of Abraham Lincoln and Boris Johnson."
```

## Options

- `max_tokens_to_sample`, default 10_000: The maximum number of tokens to generate before stopping

Use like this:
```bash
llm -m claude -o max_tokens_to_sample 20 "Sing me the alphabet"
 Here is the alphabet song:

A B C D E F G
H I J
```

## Development

To set up this plugin locally, first checkout the code. Then create a new virtual environment:

    cd llm-claude
    python3 -m venv venv
    source venv/bin/activate

Now install the dependencies and test dependencies:

    pip install -e '.[test]'

To run the tests:

    pytest
