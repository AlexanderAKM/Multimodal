# for network in language random none;
# do
#     python -m generate_lesion \
#         --model-name gpt2 \
#         --prompt "The quick brown fox" \
#         --percentage 5 \
#         --network $network
# done 

for network in language random none;
do
    python -m generate_lesion \
        --model-name llava-hf/llava-1.5-7b-hf \
        --prompt "The quick brown fox" \
        --percentage 5 \
        --network $network
done 