set +ex

RED='\033[31m'
RESET='\033[0m'

rm -rf pip-tmp/

# first check we still need to vendor
pip install anthropic \
    --target=./pip-tmp --no-deps --upgrade

if ! grep -'Requires-Dist: pydantic (.*<2.0.0)' pip-tmp/anthropic-*.dist-info/METADATA; then
    echo
    echo "${RED}pydantic pin not found. Unvendor anthropic?"
    echo ${RESET}
    exit 1
fi

rm -rf pip-tmp/

# fetch forked library, with unpinned pydantic
pip install https://github.com/tomviner/anthropic-sdk-python/archive/pydantic-v2-compat.zip \
    --target=./pip-tmp --no-deps --upgrade

# include the licence and metadata
mv pip-tmp/anthropic-*.dist-info/LICENSE pip-tmp/anthropic
mv pip-tmp/anthropic-*.dist-info/METADATA pip-tmp/anthropic

rm -rf llm_claude/vendored_anthropic/
mv pip-tmp/anthropic/ llm_claude/vendored_anthropic/

rm -rf pip-tmp/
