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
    print(ds[0:5])
    print(ds["label"][:2])

    predictions = []
    references = ds["label"][:1]

    for i in range(2, 3):
        
        text = ds["premise"][i-2] + " " + ds["hypothesis"][i-2] + " " + str(ds["label"][i-2]) + "\n" + \
               ds["premise"][i-1] + " " + ds["hypothesis"][i-1] + " " + str(ds["label"][i-1]) + "\n" + \
               ds["premise"][i] + " " + ds["hypothesis"][i] + "\n" 

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

    predictions[0] = int(predictions[0])
    # Wrong, computing metric between text and integer...
    results = glue_metric.compute(predictions=predictions, references=references)
    print(results)

if __name__ == "__main__":
    run_eval(None, None, None)

    # wikitext103