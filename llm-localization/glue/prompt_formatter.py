"""
Prompt formatting utilities for GLUE benchmark tasks.
These functions convert dataset examples into text prompts suitable for LLM evaluation.
"""

from typing import Dict

def format_prompt(task_name: str, example: Dict) -> str:
    """Format a GLUE example as a text prompt for LLMs.
    
    Args:
        task_name: Name of the GLUE task
        example: Example from the GLUE dataset
        
    Returns:
        Formatted prompt for LLMs
    """
    if task_name == "cola":
        return f"Sentence: {example['sentence']}\nIs this sentence grammatically correct? Answer:"
    
    elif task_name in ["mnli", "mnli-mm"]:
        return f"Premise: {example['premise']}\nHypothesis: {example['hypothesis']}\nDoes the premise entail the hypothesis? Answer with 'entailment', 'neutral', or 'contradiction':"
    
    elif task_name == "mrpc":
        return f"Sentence 1: {example['sentence1']}\nSentence 2: {example['sentence2']}\nAre these sentences paraphrases of each other? Answer:"
    
    elif task_name == "qnli":
        return f"Question: {example['question']}\nSentence: {example['sentence']}\nDoes the sentence answer the question? Answer:"
    
    elif task_name == "qqp":
        return f"Question 1: {example['question1']}\nQuestion 2: {example['question2']}\nAre these questions asking the same thing? Answer:"
    
    elif task_name == "rte":
        return f"Premise: {example['premise']}\nHypothesis: {example['hypothesis']}\nDoes the premise entail the hypothesis? Answer:"
    
    elif task_name == "sst2":
        return f"Review: {example['sentence']}\nSentiment (positive/negative):"
    
    elif task_name == "stsb":
        return f"Sentence 1: {example['sentence1']}\nSentence 2: {example['sentence2']}\nOn a scale from 0 to 5, how similar are these sentences?"
    
    elif task_name == "wnli":
        return f"Sentence 1: {example['sentence1']}\nSentence 2: {example['sentence2']}\nIs sentence 2 entailed by sentence 1? Answer:"
    
    else:
        raise ValueError(f"Task {task_name} not supported")