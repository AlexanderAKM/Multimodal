
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

for network in language random none;
do
    $python -m generate_lesion \
        --model-name gpt2 \
        --prompt "The quick brown fox" \
        --percentage 0.1 \
        --localize-range 100-100 \
        --network $network
done 

# do
#     $python -m generate_lesion \
#         --model-name llava-hf/llava-1.5-7b-hf \
#         --prompt "The quick brown fox" \
#         --percentage 0.1 \
#         --localize-range 100-100 \
#         --network $network
# done 
