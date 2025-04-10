"""
GLUE Benchmark evaluation for language models.
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
from tqdm import tqdm

from glue.evaluator import run_eval, run_glue_benchmark
from utils import get_log_likelihood


def run_blimp(model, tokenizer, device, num_examples=100):
    """
    Runs a BLiMP evaluation. Each example is assumed to have a pair of sentences:
    one grammatical ('sentence_good') and one ungrammatical ('sentence_bad').
    The model’s log likelihood is computed for each sentence; if the grammatical sentence
    is assigned a higher likelihood, the example is counted as correct.
    """
    print("Running BLiMP evaluation...")
    try:
        ds = load_dataset("nyu-mll/blimp", "adjunct_island", split="train")
        ds = ds.select(range(num_examples))  # Limit to num_examples
    except Exception as e:
        print("BLiMP dataset not available. Skipping BLiMP evaluation.")
        print(e)
        return
    correct = 0
    total = 0
    for example in tqdm(ds, desc="Evaluating BLiMP examples", unit="example"):
        # Adjust the field names if your BLiMP dataset uses different keys.
        if "sentence_good" in example and "sentence_bad" in example:
            good = example["sentence_good"]
            bad = example["sentence_bad"]
        elif "good" in example and "bad" in example:
            good = example["good"]
            bad = example["bad"]
        else:
            continue
        ll_good = get_log_likelihood(good, model, tokenizer, device)
        ll_bad = get_log_likelihood(bad, model, tokenizer, device)
        if ll_good > ll_bad:
            correct += 1
        total += 1
    acc = correct / total if total > 0 else 0
    print("BLiMP accuracy:", acc)


def run_syntaxgym(model, tokenizer, device, num_examples=100):
    """
    Runs a SyntaxGym evaluation. The assumed structure is similar to BLiMP,
    where each test item contains a grammatical sentence and a comparable
    ungrammatical sentence. (Modify as needed to fit the actual SyntaxGym data.)
    """
    print("Running SyntaxGym evaluation...")
    try:
        ds = load_dataset("syntaxgym")
        ds = ds.select(range(num_examples))  # Limit to num_examples
    except ImportError:
        print("SyntaxGym dataset not available. Please install the dataset.")
        return
    except Exception as e:
        print("SyntaxGym dataset not available. Skipping SyntaxGym evaluation.")
        return
    correct = 0
    total = 0
    for example in tqdm(ds, desc="Evaluating SyntaxGym examples", unit="example"):
        if "sentence_good" in example and "sentence_bad" in example:
            good = example["sentence_good"]
            bad = example["sentence_bad"]
        else:
            continue
        ll_good = get_log_likelihood(good, model, tokenizer, device)
        ll_bad = get_log_likelihood(bad, model, tokenizer, device)
        if ll_good > ll_bad:
            correct += 1
        total += 1
    acc = correct / total if total > 0 else 0
    print("SyntaxGym accuracy:", acc)


def run_glue(
    model,
    tokenizer,
    processor,
    device="cpu",
):
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
        task_name="mnli",
        batch_size=1,
        num_examples=10,  # Limit for faster testing
        save_dir="./glue_results",
    )

    # Uncomment to run full GLUE benchmark
    # all_results = run_glue_benchmark(
    #     model=model,
    #     tokenizer=tokenizer,
    #     device=device,
    #     batch_size=8,
    #     num_examples=10,  # Limit for faster testing
    #     save_dir="./glue_results"
    # )


if __name__ == "__main__":
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "mps" if torch.backends.mps.is_available() else device
    model.to(device)

    # Run GLUE evaluation
    # run_glue(
    #     model,
    #     tokenizer,
    #     processor=None,
    #     device=device,
    #     task_name="mnli",
    #     batch_size=1,
    #     num_examples=10,
    # )

    # Run BLiMP evaluation
    run_blimp(model, tokenizer, device, num_examples=10)

    # Run SyntaxGym evaluation
    run_syntaxgym(model, tokenizer, device, num_examples=10)
