set +ex

pip install https://github.com/tomviner/anthropic-sdk-python/archive/pydantic-v2-compat.zip \
    --target=./pip-tmp --no-deps
mv pip-tmp/anthropic-*.dist-info/LICENSE pip-tmp/anthropic

rm -rf llm_claude/vendored_anthropic/
mv pip-tmp/anthropic/ llm_claude/vendored_anthropic/

rm -rf pip-tmp/
