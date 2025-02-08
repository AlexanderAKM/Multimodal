
cd llm-localization


# determine if python3 is installed otherwise use python
if command -v python3 &> /dev/null
then
    echo "using python3"
    python=python3

else
    echo "using python"
    python=python
fi

$python localize.py  \
 --model-name openai-community/gpt2 \
 --percentage 0.1 \
 --network language \
 --localize-range 100-100 \
 --pooling last-token \
