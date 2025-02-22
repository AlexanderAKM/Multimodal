from datasets import load_dataset
from evaluate import load
import torch

# from transformers import PyTorchBenchmark, PyTorchBenchmarkArguments
# args = PyTorchBenchmarkArguments(models=["google-bert/bert-base-uncased"], batch_sizes=[8], sequence_lengths=[8, 32, 128, 512])
# benchmark = PyTorchBenchmark(args)

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
