for network in language random none;
do
    python -m generate_lesion \
        --model-name gpt2 \
        --prompt "The quick brown fox" \
        --percentage 5 \
        --network $network
done 