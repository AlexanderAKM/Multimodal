"""
Core evaluation functionality for GLUE benchmark
"""

from datasets import load_dataset
from evaluate import load
import torch
import numpy as np
from tqdm import tqdm
import os

from .config import GLUE_TASKS
from .prompt_formatter import format_prompt
from .response_parser import parse_generated_text
from .utils import setup_logging, save_results

# Set up logging
logger = setup_logging()

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
    """Run GLUE evaluation on a generative model.
    
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
        
        # Process each example in the batch INDIVIDUALLY
        for j in range(len(batch["label"])):
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
        save_path = save_results(results, save_dir, task_name)
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