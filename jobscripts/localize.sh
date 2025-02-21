#!/bin/bash
#SBATCH --job-name=localize
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --mem=64GB
#SBATCH --gpus-per-node=1

cd /home1/s5193400/Multimodal/llm-localization


module purge
source $HOME/venvs/multimodal/bin/activate
module load SciPy-bundle/2023.07-gfbf-2023a
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load Python/3.11.5-GCCcore-13.2.0

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
 --model-name llava-hf/llava-1.5-7b-hf \
 --percentage 0.1 \
 --network language \
 --localize-range 100-100 \
 --pooling last-token \



