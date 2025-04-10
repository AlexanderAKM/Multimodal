from datasets import load_dataset
from evaluate import load
import torch
import numpy as np
from tqdm import tqdm
import logging
import os
from typing import Dict, List, Optional, Union, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# GLUE task configurations
GLUE_TASKS = {
    "cola": {
        "metric": "matthews_correlation",
        "num_labels": 2,
    },
    "mnli": {
        "metric": "accuracy",
        "num_labels": 3,
    },
    "mrpc": {
        "metric": "f1",
        "num_labels": 2,
    },
    "qnli": {
        "metric": "accuracy",
        "num_labels": 2,
    },
    "qqp": {
        "metric": "f1",
        "num_labels": 2,
    },
    "rte": {
        "metric": "accuracy",
        "num_labels": 2,
    },
    "sst2": {
        "metric": "accuracy",
        "num_labels": 2,
    },
    "stsb": {
        "metric": "spearmanr",
        "num_labels": 1,
    },
    "wnli": {
        "metric": "accuracy",
        "num_labels": 2,
    }
}

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
        return f"Sentence 1: {example['sentence1']}\nSentence 2: {example['sentence2']}\nAre these sentences paraphrases of each other? Answer with 'yes' or 'no':"
    elif task_name == "qnli":
        return f"Question: {example['question']}\nSentence: {example['sentence']}\nDoes the sentence answer the question? Answer with 'yes' or 'no':"
    elif task_name == "qqp":
        return f"Question 1: {example['question1']}\nQuestion 2: {example['question2']}\nAre these questions asking the same thing? Answer with 'yes' or 'no':"
    elif task_name == "rte":
        return f"Sentence 1: {example['sentence1']}\nSentence 2: {example['sentence2']}\nDoes sentence 1 entail sentence 2? Answer with 'yes' or 'no':"
    elif task_name == "sst2":
        return f"Review: {example['sentence']}\nSentiment: Positive or Negative? Answer with '1' for positive and '0' for negative:"
    elif task_name == "stsb":
        return f"Sentence 1: {example['sentence1']}\nSentence 2: {example['sentence2']}\nOn a scale from 0 to 5, how similar are these sentences?"
    elif task_name == "wnli":
        return f"Sentence 1: {example['sentence1']}\nSentence 2: {example['sentence2']}\nIs sentence 2 entailed by sentence 1? Answer with 'yes' or 'no':"
    else:
        raise ValueError(f"Task {task_name} not supported")

def parse_generated_text(task_name: str, text: str) -> Union[int, float]:
    """Parse generated text to extract prediction.
    Args:
        task_name: Name of the GLUE task
        text: Generated text from the model
    Returns:
        Parsed prediction as an integer or float
    """
    text = text.strip().lower()
    
    if task_name == "cola":
        if "yes" in text or "correct" in text or "acceptable" in text or "grammatical" in text:
            return 1
        else:
            return 0
    
    elif task_name in ["mnli", "mnli-mm"]:
        if "entailment" in text:
            return 2
        elif "contradiction" in text:
            return 0
        else:  # neutral or anything else
            return 1
    
    elif task_name in ["mrpc", "qqp"]:
        if "yes" in text or "equivalent" in text or "paraphrase" in text or "same" in text:
            return 1
        else:
            return 0
    
    elif task_name == "qnli":
        if "yes" in text or "entail" in text:
            return 0  # entailment
        else:
            return 1  # not entailment
    
    elif task_name == "rte":
        if "entail" in text or "yes" in text:
            return 0  # entailment
        else:
            return 1  # not entailment
    
    elif task_name == "sst2":
        if "positive" in text:
            return 1
        else:
            return 0
    
    elif task_name == "stsb":
        # Try to extract a number from the text
        try:
            for word in text.split():
                if word.replace('.', '').isdigit():
                    score = float(word)
                    # Normalize to 0-5 range if needed
                    if score > 5:
                        score = 5.0
                    if score < 0:
                        score = 0.0
                    return score
            # Default to middle value if no number found
            return 2.5
        except ValueError:
            return 2.5
    
    elif task_name == "wnli":
        if "yes" in text or "entail" in text:
            return 1
        else:
            return 0
    
    else:
        raise ValueError(f"Task {task_name} not supported")

def run_eval(
    model, 
    tokenizer=None, 
    processor=None, 
    device="cpu",
    task_name="mnli",
    batch_size=8,
    num_examples=None,
    save_dir=None
):
    """Run GLUE evaluation on a generative model like GPT-2.
    
    Args:
        model: The model to evaluate
        tokenizer: The tokenizer (for non-multimodal models)
        processor: The processor (for multimodal models)
        device: Device to run evaluation on
        task_name: GLUE task name (e.g., "mnli", "cola", etc.)
        batch_size: Batch size for evaluation
        num_examples: Maximum number of examples to evaluate (None = all)
        save_dir: Directory to save results to (None = don't save)
        
    Returns:
        Dict of evaluation results
    """
    assert model is not None, "Model must be provided."
    assert tokenizer is not None or processor is not None, "Tokenizer or processor must be provided."
    
    # For mnli-mm, we use the mismatched validation set
    if task_name == "mnli-mm":
        dataset_name = "mnli_mismatched"
    elif task_name == "mnli":
        dataset_name = "mnli_matched"
    else:
        dataset_name = task_name
    
    # Load dataset
    logger.info(f"Loading {task_name} dataset...")
    ds = load_dataset("nyu-mll/glue", dataset_name, split="validation")
    if num_examples is not None:
        ds = ds.select(range(min(num_examples, len(ds))))
    
    # Load metric
    logger.info(f"Loading {task_name} metric...")
    metric = load("glue", task_name)
    
    # Prepare for evaluation
    predictions = []
    references = ds["label"]
    
    # Process examples in batches
    num_batches = (len(ds) + batch_size - 1) // batch_size
    logger.info(f"Evaluating {len(ds)} examples in {num_batches} batches...")
    
    for i in tqdm(range(0, len(ds), batch_size)):
        batch = ds[i:i + batch_size]
        batch_predictions = []

        print(batch)
        
        # Process each example in the batch INDIVIDUALLY
        for j in range(len(batch["label"])):
            # Skip if the example is empty
            if all(batch[key][j] is None for key in batch.keys()):
                continue

            # Extract a single example from the batch
            example = {}
            for key in batch.keys():
                if key in batch:
                    example[key] = batch[key][j]
            
            # Format the prompt for this specific example
            prompt = format_prompt(task_name, example)
            
            # Tokenize the prompt (for a single example)
            if processor is not None:
                inputs = processor(images=None, text=prompt, return_tensors="pt").to(device)
            else:
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            # Generate text
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=20,  # Limit generation to a reasonable length
                    do_sample=False,    # Use greedy decoding for evaluation
                    pad_token_id=tokenizer.eos_token_id if tokenizer is not None else None
                )
            
            # Decode the generated text
            if processor is not None:
                generated_text = processor.decode(outputs[0], skip_special_tokens=True)
            else:
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Remove the input prompt from the generated text
                if generated_text.startswith(prompt):
                    generated_text = generated_text[len(prompt):]
            
            # Parse the generated text to get the prediction
            prediction = parse_generated_text(task_name, generated_text)
            batch_predictions.append(prediction)
            
            # Log example for debugging (every 50 examples)
            if (i + j) % 50 == 0:
                logger.info(f"Example {i + j}:")
                logger.info(f"Prompt: {prompt}")
                logger.info(f"Generated: {generated_text}")
                logger.info(f"Prediction: {prediction}, Reference: {example['label']}")
        
        # Add batch predictions to overall predictions
        predictions.extend(batch_predictions)
    
    # Validate predictions and references
    assert len(predictions) == len(references), f"Got {len(predictions)} predictions but {len(references)} references"
    
    # Convert predictions to numpy array if needed
    predictions = np.array(predictions)
    
    # Compute metrics
    logger.info("Computing metrics...")
    results = metric.compute(predictions=predictions, references=references)
    
    # Print results
    for metric_name, value in results.items():
        logger.info(f"{task_name} - {metric_name}: {value:.4f}")
    
    # Save results if requested
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{task_name}_results.txt")
        with open(save_path, "w") as f:
            for metric_name, value in results.items():
                f.write(f"{task_name} - {metric_name}: {value:.4f}\n")
        logger.info(f"Results saved to {save_path}")
    
    return results

def run_glue_benchmark(
    model, 
    tokenizer=None, 
    processor=None, 
    device="cpu",
    tasks=None,
    batch_size=8,
    num_examples=None,
    save_dir=None
):
    """Run full GLUE benchmark on multiple tasks.
    
    Args:
        model: The model to evaluate
        tokenizer: The tokenizer (for non-multimodal models)
        processor: The processor (for multimodal models)
        device: Device to run evaluation on
        tasks: List of GLUE tasks to run (None = all tasks)
        batch_size: Batch size for evaluation
        num_examples: Maximum number of examples to evaluate per task (None = all)
        save_dir: Directory to save results to (None = don't save)
        
    Returns:
        Dict of evaluation results for all tasks
    """
    if tasks is None:
        tasks = list(GLUE_TASKS.keys())
    
    all_results = {}
    
    for task in tasks:
        logger.info(f"Running evaluation on {task}...")
        results = run_eval(
            model=model,
            tokenizer=tokenizer,
            processor=processor,
            device=device,
            task_name=task,
            batch_size=batch_size,
            num_examples=num_examples,
            save_dir=save_dir
        )
        all_results[task] = results
    
    # Compute average score across all tasks
    avg_score = 0.0
    count = 0
    
    for task, results in all_results.items():
        metric_name = GLUE_TASKS[task]["metric"]
        if metric_name in results:
            avg_score += results[metric_name]
            count += 1
    
    if count > 0:
        avg_score /= count
        logger.info(f"Average GLUE score: {avg_score:.4f}")
        
        # Save average score if requested
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, "glue_average.txt"), "w") as f:
                f.write(f"Average GLUE score: {avg_score:.4f}\n")
    
    return all_results

if __name__ == "__main__":
    # Example usage with GPT-2
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    # Set pad token to eos token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    # Run evaluation on a single task with limited examples
    results = run_eval(
        model=model,
        tokenizer=tokenizer,
        device=device,
        task_name="wnli",
        batch_size=2,
        num_examples=4,  # Limit for faster testing
        save_dir="./glue_results"
    )
    
    # Uncomment to run full GLUE benchmark
    # all_results = run_glue_benchmark(
    #     model=model,
    #     tokenizer=tokenizer,
    #     device=device,
    #     batch_size=2,
    #     num_examples=4,  # Limit for faster testing
    #     save_dir="./glue_results"
    # )
