import os
import torch
import argparse
import numpy as np
from transformers import AutoTokenizer
from transformers import AutoProcessor

from models.modeling_gpt2 import GPT2LMHeadModel
from models.modeling_llama import LlamaForCausalLM
from models.modeling_phi3 import Phi3ForCausalLM
from models.modeling_gemma import GemmaForCausalLM
# from models.modeling_falcon import FalconForCausalLM
from models.modeling_mistral import MistralForCausalLM
from models.modeling_llava import LlavaForConditionalGeneration

from eval import run_eval, get_log_likelihood, run_blimp, run_glue, run_syntaxgym

# Directory for cached model data
CACHE_DIR = os.environ.get("LOC_CACHE", "cache")

if __name__ == "__main__":
    """
    Script to load a specified language model, apply a localization mask, and generate text output.

    Arguments:
        --model-name (str, required): The name of the model to be loaded.
        --prompt (str, required): The input prompt for text generation.
        --percentage (float, required): The percentage of active units for localization masking.
        --network (str, optional): Type of network mask to apply (default: 'language').
        --device (str, optional): Device for model execution ('cuda' or 'cpu').
        --seed (int, optional): Random seed for reproducibility (default: 42).
        --pooling (str, optional): Pooling method, either 'last-token' or 'mean' (default: 'last-token').
        --localize-range (str, optional): Range specification for localization (default: '100-100').
    """
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model-name", type=str, required=True)
    argparser.add_argument("--prompt", type=str, required=True)
    argparser.add_argument("--percentage", type=float, required=True)
    argparser.add_argument("--network", type=str, default="language", choices=["language", "random", "none"])
    argparser.add_argument("--device", type=str, default=None)
    argparser.add_argument("--seed", type=int, default=42)
    argparser.add_argument("--pooling", type=str, default="last-token", choices=["last-token", "mean"])
    argparser.add_argument("--localize-range", type=str, default="100-100")

    args = argparser.parse_args()

    # Set up configurations
    seed = args.seed
    percentage = args.percentage


    # Determine device: CUDA, CPU, or MPS
    device = "cpu" # Default to CPU

    if torch.cuda.is_available():
        device = "cuda"

    if torch.backends.mps.is_available():
        device = "mps"

    # override device if specified
    if args.device is not None:
        device = args.device

    model_name = args.model_name
    network = args.network
    prompt = args.prompt
    pooling = args.pooling
    loc_range = args.localize_range

    print(f"> Running with model {model_name}")

    # Load the specified model
    if "gpt2" in model_name:
        model = GPT2LMHeadModel.from_pretrained(model_name)
    elif "Llama" in model_name:
        model = LlamaForCausalLM.from_pretrained(model_name)
    elif "Phi" in model_name:
        model = Phi3ForCausalLM.from_pretrained(model_name)
    elif "gemma" in model_name:
        model = GemmaForCausalLM.from_pretrained(model_name)
    elif "falcon" in model_name:
        model = FalconForCausalLM.from_pretrained(model_name)
    elif "Mistral" in model_name:
        model = MistralForCausalLM.from_pretrained(model_name)
    elif "llava" in model_name:
        model = LlavaForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map=device
            )
    else:
        raise ValueError(f"Model {model_name} not supported")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    processor = None
    if "llava" in model_name:
        processor = AutoProcessor.from_pretrained(model_name)

    model.to(device)
    model.eval()
    # print(model)

    # print(f"> Running with {network} mask")

    # Define the mask path
    if network in ["language", "random"]:
        mask_path = f"{model_name.split('/')[-1]}_network=language_pooling={pooling}_range={loc_range}_perc={percentage}_nunits=None_pretrained=True.npy"
    else:
        mask_path = None
    print(f"{CACHE_DIR}//{mask_path}")
    # Load and apply the language mask
    if mask_path is not None:
        language_mask = np.load(f"{CACHE_DIR}/{mask_path}")
        num_active_units = int(language_mask.sum())

        if network == "random":
            num_layers, hidden_dim = language_mask.shape
            total_num_units = np.prod(language_mask.shape)
            invlang_mask_indices = np.arange(total_num_units)[(1 - language_mask).flatten().astype(bool)]
            np.random.seed(seed)
            rand_indices = np.random.choice(invlang_mask_indices, size=num_active_units, replace=False)
            lang_mask_rand = np.full(total_num_units, 0)
            lang_mask_rand[rand_indices] = 1
            assert np.sum(lang_mask_rand) == num_active_units
            language_mask = lang_mask_rand.reshape((num_layers, hidden_dim))
        print("Loaded language mask with", num_active_units, "units, with shape", language_mask.shape)

        model.set_language_selective_mask(torch.tensor(language_mask).to(device)) # trouble
    else:
        model.set_language_selective_mask(None)

    print("Language mask applied")

    # QUALIATIVELY EVALUATE
    # Prepare inputs and generate text
    # inputs = tokenizer(prompt, return_tensors="pt").to(device)
    # outputs = model.generate(
    #     **inputs,
    #     max_length=20, # original 20
    #     do_sample=True,
    #     temperature=0.7,
    #     num_return_sequences=1,
    # )

    # print(tokenizer.decode(outputs[0], skip_special_tokens=True))


    # Benchmarks

    run_glue(model, tokenizer, processor, device)
    #run_blimp(model, tokenizer, device)
    #run_syntaxgym(model, tokenizer, device)