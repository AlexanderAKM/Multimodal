
cd llm-localization
 python localize.py  \
 --model-name openai-community/gpt2 \
 --percentage 0.1 \
 --network language \
 --localize-range 100-100 \
 --pooling last-token \
