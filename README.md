# Multimodal
Project of identifying causally task-relevant units in multi-modal models. Builds up on [2411.02280] The LLM Language Network: A Neuroscientific Approach for Identifying Causally Task-Relevant Units. 

TODO:
- fix cache issue
    - ``localize.py`` works in the first time, but after caching, it does not work.


# Important bits methodology
- "We then capture the activations from the units at the output of each Transformer block for each stimulus. We define the model’s language network as the top-k units that maximize the difference in activation magnitude between sentences and strings of non-words, measured by positive t-values from a Welch’s t-test.
- They define a "unit" to be each dimension after a Transformer block.
- They do 10 models, we should do 2/3?
- 