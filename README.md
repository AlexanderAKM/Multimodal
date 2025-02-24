# Multimodal
Project of identifying causally task-relevant units in multi-modal models. Builds up on [2411.02280] The LLM Language Network: A Neuroscientific Approach for Identifying Causally Task-Relevant Units. 

## Setup

First clone the repository and install the requirements.

```bash
conda create -n multimodal python=3.10
conda activate multimodal
pip install -r requirements.txt
```


## TODO

TODO:

0. Gpt2 instead of large VLLM, for faster development
1. We change generate_lesion.py so that it runs on different benchmarks and saves performance.
    - Glue, Blimp, SyntaxGym


1. Get compute from AISIG?
    - Rerun bash localize lesions for full dataset instead of 5.

2. We change generate_lesion.py so that it runs on benchmark and saves performance.
    - Glue, Blimp, SyntaxGym
    - some vision dataset?
    - BATCHING???

3. We find/create vision dataset of images vs random pixels. 
    - Create proper Class for it.
    - Should be able to run llava on it.

4. After vision dataset, run localize lesions on it.


4. Use GPT for making proper eval function!

#### Later stuff

- We should read something about benchmarks.
- Need to actually understand the language dataset, not hugely important, but for writing paper prob.



# Important bits methodology
- "We then capture the activations from the units at the output of each Transformer block for each stimulus. We define the model’s language network as the top-k units that maximize the difference in activation magnitude between sentences and strings of non-words, measured by positive t-values from a Welch’s t-test.
- They define a "unit" to be each dimension after a Transformer block.
- They do 10 models, we should do 2/3?
- They ablate top-{0.125, 0.25, 0.5, 1}% of language-selective units so we can do this similarly for all tasks.
- Then they look at similarity between the language network in LLMs and Brains
    - They have four conditions based (matrix lexical vs syntactical)
    - In neuroscience the normal sentences show a lot more activation than the other three.
    - This is similar int he language networks in LLMs.
- They do some pearson correlation between the predicted brain activity of people on two neuroscience datasets.
    - With a percentage of units (language vs random) they predict brain activity doing ridge regression
    - Some significance but I'm not too sure what this shows.
- I don't quite see the point in the above (although it is interesting). Rather, we could do **feature visualization**!!
    - They only show the distribution of significant units in the respective layers, but it would be way cooler if we
    - actually visualized it. This would be similar to fMRI!
- Our paper should focus more on *AI alignment* and how we approach this through *neuralignment*.
    - If we manage to show these specializations in the architecture, that is quite a find for interpretability of the model.

- Look at CHIP (Conflict avoidance, Higher capacity, Increased processing Speed, Parallel processing) relating lateralization/specialization in human brain.

- "No one in neuroscience has tried vision vs. non-vision :)"
- Should definitely look at the model architecture in more depth and see where the language and vision units should be expected to be identified.
    
