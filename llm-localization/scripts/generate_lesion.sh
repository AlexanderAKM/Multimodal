# for network in language random none;
# do
#     python -m generate_lesion \
#         --model-name gpt2 \
#         --prompt "The quick brown fox" \
#         --percentage 5 \
#         --network $network
# done 
# LOC_CACHE="C:Users\User\Personal Projects\Multimodal\llm-localization\cache"
# cd llm-localization/
for network in language random none;
do
    python -m generate_lesion \
        --model-name llava-hf/llava-1.5-7b-hf \
        --prompt "The quick brown fox" \
        --percentage 0.1 \
        --network $network
done 