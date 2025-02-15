from evaluate import load
from datasets import load_dataset
import torch

# from transformers import PyTorchBenchmark, PyTorchBenchmarkArguments
# args = PyTorchBenchmarkArguments(models=["google-bert/bert-base-uncased"], batch_sizes=[8], sequence_lengths=[8, 32, 128, 512])
# benchmark = PyTorchBenchmark(args)

def run_eval(model, processor, device="cpu"):
    ds = load_dataset("nyu-mll/glue", "mnli_matched", split="validation").select(range(1000))
    glue_metric = load("glue", "cola")
    print(ds)

    predictions = []
    references = ds["label"]

    for i in range(1000):
        # give more context (2 shot?)
        text = ds["premise"][i] + " " + ds["hypothesis"][i]

        inputs = processor(images=None, text=text, return_tensors="pt").to(device, dtype=torch.float16)

        outputs = model.generate(
            **inputs,
            max_length=len(text) // 2, # original 20
            do_sample=True,
            temperature=0.7,
            num_return_sequences=1,
        )

        outputs = processor.decode(outputs[0], skip_special_tokens=True)
        print(text)
        print(outputs)
        predictions.append(outputs)
        
    assert len(predictions) == len(references), "Predictions and references must be the same length."


    results = glue_metric.compute(predictions=predictions, references=references)


if __name__ == "__main__":
    run_eval(None, None, None)

    # wikitext103