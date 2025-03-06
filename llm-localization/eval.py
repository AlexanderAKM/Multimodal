from datasets import load_dataset
from evaluate import load
import torch

# from transformers import PyTorchBenchmark, PyTorchBenchmarkArguments
# args = PyTorchBenchmarkArguments(models=["google-bert/bert-base-uncased"], batch_sizes=[8], sequence_lengths=[8, 32, 128, 512])
# benchmark = PyTorchBenchmark(args)


def get_log_likelihood(text, model, tokenizer, device):
    """
    Compute the total log likelihood of a given text.
    (This multiplies the per-token loss by the number of tokens.)
    """
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    # outputs.loss is averaged over tokens
    log_likelihood = -outputs.loss.item() * inputs["input_ids"].shape[1]
    return log_likelihood

def run_glue(model, tokenizer, processor, device, num_examples=100):
    """
    Runs a prompt-based evaluation on GLUE’s MNLI task.
    For each example, a prompt is constructed using the premise and hypothesis.
    A simple keyword matching rule is used to map the generated text
    to one of the three labels.
    """
    print("Running GLUE evaluation (MNLI)...")
    ds = load_dataset("glue", "mnli", split=f"validation[:{num_examples}]")
    # Mapping for MNLI labels: 0 -> contradiction, 1 -> neutral, 2 -> entailment.
    label_map = {0: "contradiction", 1: "neutral", 2: "entailment"}
    correct = 0
    total = 0
    for example in ds:
        premise = example["premise"]
        hypothesis = example["hypothesis"]
        true_label = label_map[example["label"]]
        prompt = (f"Premise: {premise}\n"
                  f"Hypothesis: {hypothesis}\n"
                  "Does the hypothesis follow from the premise? Answer:")
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_length=50)
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True).lower()
        # Simple keyword matching to extract a prediction
        if "entail" in output_text:
            pred_label = "entailment"
        elif "contradict" in output_text:
            pred_label = "contradiction"
        elif "neutral" in output_text:
            pred_label = "neutral"
        else:
            # Default if nothing obvious is found
            pred_label = "neutral"
        if pred_label == true_label:
            correct += 1
        total += 1
    acc = correct / total if total > 0 else 0
    print("GLUE (MNLI) accuracy:", acc)

def run_blimp(model, tokenizer, device, num_examples=100):
    """
    Runs a BLiMP evaluation. Each example is assumed to have a pair of sentences:
    one grammatical ('sentence_good') and one ungrammatical ('sentence_bad').
    The model’s log likelihood is computed for each sentence; if the grammatical sentence
    is assigned a higher likelihood, the example is counted as correct.
    """
    print("Running BLiMP evaluation...")
    try:
        ds = load_dataset("blimp", split=f"validation[:{num_examples}]")
    except Exception as e:
        print("BLiMP dataset not available. Skipping BLiMP evaluation.")
        return
    correct = 0
    total = 0
    for example in ds:
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
        ds = load_dataset("syntaxgym", split=f"validation[:{num_examples}]")
    except Exception as e:
        print("SyntaxGym dataset not available. Skipping SyntaxGym evaluation.")
        return
    correct = 0
    total = 0
    for example in ds:
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

def run_eval(model, tokenizer=None, processor=None, device="cpu"):
    assert model is not None, "Model must be provided."
    assert tokenizer is not None or processor is not None, "Tokenizer or processor must be provided."


    ds = load_dataset("nyu-mll/glue", "mnli_matched", split="validation").select(range(1000))
    glue_metric = load("glue", "cola")

    predictions = []
    references = ds["label"][:1]

    for i in range(2, 3):
        text = (
            f"{ds['premise'][i-2]}  {ds['hypothesis'][i-2]} {str(ds['label'][i-2])}\n"
            f"{ds['premise'][i-1]} {ds['hypothesis'][i-1]} {str(ds['label'][i-1])}\n"
            f"{ds['premise'][i]} {ds['hypothesis'][i]}"
        )

        if processor is not None:
            inputs = processor(images=None, text=text, return_tensors="pt").to(device, dtype=torch.float16)
        else:
            inputs = tokenizer(text, return_tensors="pt").to(device)

        outputs = model.generate(
            **inputs,
            max_length=len(text) // 2, # original 20
            do_sample=True,
            temperature=0.7,
            num_return_sequences=1,
        )


        detokenize_config = {"skip_special_tokens": True}
        
        predictions.append(outputs[0]) # raw token ids

        if processor is not None:
            outputs = processor.decode(outputs[0], **detokenize_config)
        else:
            outputs = tokenizer.decode(outputs[0], **detokenize_config)


        print()
        print("="*80)
        print(f"Example {i}:")
        print(text)
        print("-"*80)
        print(outputs)
        print("="*80)
        print()

        
    assert len(predictions) == len(references), "Predictions and references must be the same length."

    # print(type(predictions[0]), predictions[0])
    # print(type(references[0]), references[0])
    # predictions[0] = int(predictions[0])
    # Wrong, computing metric between text and integer...

    print("Predictions:", predictions)
    print("References:", references)

    # {'predictions': Value(dtype='int64', id=None), 'references': Value(dtype='int64', id=None)}
    results = glue_metric.compute(predictions=predictions[0], references=references)
    print(results)

if __name__ == "__main__":
    run_eval(None, None, None)

    # wikitext103
