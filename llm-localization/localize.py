from typing import List, Dict

import os
import torch
import argparse
import numpy as np
import transformers

from tqdm import tqdm
from torch.utils.data import DataLoader
from scipy.stats import ttest_ind, false_discovery_control

from model_utils import get_layer_names, get_hidden_dim
from utils import setup_hooks
from datasets import LangLocDataset, TOMLocDataset, MDLocDataset

from transformers import AutoProcessor, LlavaForConditionalGeneration

# Directory for cached language masks
CACHE_DIR = os.environ.get("LOC_CACHE", f"cache")


def extract_batch(
    model: torch.nn.Module, 
    input_ids: torch.Tensor, 
    attention_mask: torch.Tensor,
    layer_names: List[str],
    pooling: str = "last-token",
):
    """
    Extracts activations from specified layers for a given batch of input tokens.

    Args:
        model (torch.nn.Module): The transformer model from which to extract activations.
        input_ids (torch.Tensor): Tokenized input IDs.
        attention_mask (torch.Tensor): Attention mask for input.
        layer_names (List[str]): List of model layer names to extract activations from.
        pooling (str): Pooling strategy - 'last-token', 'mean', or 'sum'.

    Returns:
        Dict[str, List[torch.Tensor]]: Dictionary mapping layer names to extracted activations.
    """

    # Initialize dictionary to store activations per layer
    batch_activations = {layer_name: [] for layer_name in layer_names}
    
    # Set up hooks to capture layer representations
    hooks, layer_representations = setup_hooks(model, layer_names)

    # Run a forward pass to compute activations (without storing gradients)
    with torch.no_grad():
        _ = model(input_ids=input_ids, attention_mask=attention_mask)

    # Process activations for each sample in the batch
    for sample_idx in range(len(input_ids)):
        for layer_idx, layer_name in enumerate(layer_names):
            if pooling == "mean":
                activations = layer_representations[layer_name][sample_idx].mean(dim=0).cpu()
            elif pooling == "sum":
                activations = layer_representations[layer_name][sample_idx].sum(dim=0).cpu()
            else:  # Default: last-token pooling
                activations = layer_representations[layer_name][sample_idx][-1].cpu()

            batch_activations[layer_name] += [activations]

    # Remove hooks after activations have been extracted
    for hook in hooks:
        hook.remove()

    return batch_activations


def extract_representations(
    network: str,
    pooling: str,
    model: torch.nn.Module,
    tokenizer: transformers.PreTrainedTokenizer,
    layer_names: List[str],
    hidden_dim: int,
    batch_size: int,
    device: torch.device,
) -> Dict[str, Dict[str, np.array]]:
    """
    Extracts hidden representations from a given model for a specified dataset.

    Args:
        network (str): The type of dataset ('language', 'theory-of-mind', or 'multiple-demand').
        pooling (str): Pooling strategy to aggregate token activations.
        model (torch.nn.Module): Transformer model.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer corresponding to the model.
        layer_names (List[str]): Layers from which to extract activations.
        hidden_dim (int): Dimensionality of hidden states.
        batch_size (int): Number of samples per batch.
        device (torch.device): Device to run computations ('cuda' or 'cpu').

    Returns:
        Dict[str, Dict[str, np.array]]: Dictionary containing positive and negative activations.
    """

    # Select the appropriate dataset based on the `network` argument
    if network == "language":
        loc_dataset = LangLocDataset()
    elif network == "theory-of-mind":
        loc_dataset = TOMLocDataset()
    elif network == "multiple-demand":
        loc_dataset = MDLocDataset()
    else:
        raise ValueError(f"Unsupported network: {network}")

    # Create a DataLoader for iterating through the dataset in batches
    langloc_dataloader = DataLoader(loc_dataset, batch_size=batch_size, num_workers=0)

    print(f"> Using Device: {device}")

    # Set model to evaluation mode and move it to the appropriate device
    model.eval()
    model.to(device)

    # Initialize storage for activations (positive and negative samples)
    final_layer_representations = {
        "positive": {layer_name: np.zeros((len(loc_dataset.positive), hidden_dim)) for layer_name in layer_names},
        "negative": {layer_name: np.zeros((len(loc_dataset.negative), hidden_dim)) for layer_name in layer_names}
    }

    # Iterate over the dataset batch-wise
    for batch_idx, batch_data in tqdm(enumerate(langloc_dataloader), total=len(langloc_dataloader)):

        # Extract positive and negative samples
        sents, non_words = batch_data

        # Tokenize samples
        if network == "language":
            sent_tokens = tokenizer(sents, truncation=True, max_length=12, return_tensors='pt').to(device)
            non_words_tokens = tokenizer(non_words, truncation=True, max_length=12, return_tensors='pt').to(device)
        else:
            sent_tokens = tokenizer(sents, padding=True, return_tensors='pt').to(device)
            non_words_tokens = tokenizer(non_words, padding=True, return_tensors='pt').to(device)

        # Extract activations from model for both positive and negative samples
        batch_real_actv = extract_batch(model, sent_tokens["input_ids"], sent_tokens["attention_mask"], layer_names, pooling)
        batch_rand_actv = extract_batch(model, non_words_tokens["input_ids"], non_words_tokens["attention_mask"], layer_names, pooling)

        # Store activations for each layer in the dataset representation
        for layer_name in layer_names:
            final_layer_representations["positive"][layer_name][batch_idx * batch_size:(batch_idx + 1) * batch_size] = torch.stack(batch_real_actv[layer_name]).numpy()
            final_layer_representations["negative"][layer_name][batch_idx * batch_size:(batch_idx + 1) * batch_size] = torch.stack(batch_rand_actv[layer_name]).numpy()

    return final_layer_representations

def localize(model_id: str,
    network: str,
    pooling: str,
    model: torch.nn.Module, 
    num_units: int, 
    tokenizer: transformers.PreTrainedTokenizer, 
    hidden_dim: int, 
    layer_names: List[str], 
    batch_size: int,
    seed: int,
    device: torch.device,
    percentage: float = None,
    localize_range: str = None,
    pretrained: bool = True,
    overwrite: bool = False,
):
    """
    Identifies and localizes selective units in a transformer model based on their statistical significance
    in a given network (e.g., language, theory-of-mind, multiple-demand).

    Args:
        model_id (str): Identifier for the model.
        network (str): Network type to analyze ('language', 'theory-of-mind', 'multiple-demand').
        pooling (str): Pooling strategy to aggregate activations ('last-token' or 'mean').
        model (torch.nn.Module): The transformer model.
        num_units (int): Number of selective units to identify.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer corresponding to the model.
        hidden_dim (int): Dimensionality of hidden states.
        layer_names (List[str]): List of layers to analyze.
        batch_size (int): Batch size for processing.
        seed (int): Random seed for reproducibility.
        device (torch.device): Device ('cuda' or 'cpu') for computations.
        percentage (float, optional): Percentage of units to select (overrides `num_units`).
        localize_range (str, optional): Percentile range for selection, e.g., '100-100' or '0-0'.
        pretrained (bool, optional): Whether to use a pretrained model.
        overwrite (bool, optional): If True, overwrite existing cached masks.

    Returns:
        np.array: Binary mask indicating localized units.
    """

    # Parse the localization range (e.g., "90-100" means top 10%)
    range_start, range_end = map(int, localize_range.split("-"))

    # Define paths for caching computed masks
    save_path = f"{CACHE_DIR}/{model_id}_network={network}_pooling={pooling}_range={localize_range}_perc={percentage}_nunits={num_units}_pretrained={pretrained}.npy"
    save_path_pvalues = f"{CACHE_DIR}/{model_id}_network={network}_pooling={pooling}_pretrained={pretrained}_pvalues.npy"

    # If the mask already exists and overwrite is False, load it from disk
    if os.path.exists(save_path) and not overwrite:
        print(f"> Loading mask from {save_path}")
        return np.load(save_path)

    # Extract activations from the model using the dataset corresponding to the network type
    representations = extract_representations(
        network=network, 
        pooling=pooling,
        model=model, 
        tokenizer=tokenizer, 
        layer_names=layer_names, 
        hidden_dim=hidden_dim, 
        batch_size=batch_size, 
        device=device,
    )

    # Initialize matrices to store t-test and p-values across layers
    p_values_matrix = np.zeros((len(layer_names), hidden_dim))
    t_values_matrix = np.zeros((len(layer_names), hidden_dim))

    # Compute statistical significance of activations in each layer
    for layer_idx, layer_name in tqdm(enumerate(layer_names), total=len(layer_names)):

        # Get absolute activation values for positive and negative samples
        positive_actv = np.abs(representations["positive"][layer_name])
        negative_actv = np.abs(representations["negative"][layer_name])

        # Perform independent t-test to compare activations
        t_values_matrix[layer_idx], p_values_matrix[layer_idx] = ttest_ind(positive_actv, negative_actv, axis=0, equal_var=False)
 
    # Helper function to get top-K elements
    def is_topk(a, k=1):
        _, rix = np.unique(-a, return_inverse=True)
        return np.where(rix < k, 1, 0).reshape(a.shape)
    
    # Helper function to get bottom-K elements
    def is_bottomk(a, k=1):
        _, rix = np.unique(a, return_inverse=True)
        return np.where(rix < k, 1, 0).reshape(a.shape)
    
    # Seed random number generator for reproducibility
    np.random.seed(seed)

    # Compute the number of selective units based on the given percentage
    if percentage is not None:
        num_units = int((percentage / 100) * hidden_dim * len(layer_names))
        print(f"> Percentage: {percentage}% --> Num Units: {num_units}")

    # If a percentile range is specified, select units within that range
    if localize_range is not None and range_start < range_end:
        range_start_val = np.percentile(t_values_matrix, range_start)
        range_end_val = np.percentile(t_values_matrix, range_end)

        # Select units within the specified percentile range
        mask_range = (t_values_matrix >= range_start_val) & (t_values_matrix <= range_end_val)
        total_num_units = np.prod(mask_range.shape)
        mask_range_indices = np.arange(total_num_units)[mask_range.flatten()]

        # Randomly select units within the range
        rand_indices = np.random.choice(mask_range_indices, size=num_units, replace=False)
        language_mask = np.full(total_num_units, 0)
        language_mask[rand_indices] = 1
        language_mask = language_mask.reshape(mask_range.shape)

        print(f"> Num units in range {range_start}-{range_end}: {language_mask.sum()}")

    # If the range is set to 0-0, select the least selective units
    elif localize_range and range_start == range_end and int(range_start) == 0:
        language_mask = is_bottomk(t_values_matrix, k=num_units)

    # Otherwise, select the most selective units
    else:
        language_mask = is_topk(t_values_matrix, k=num_units)

    print(f"> Num units: {language_mask.sum()}")

    # Apply False Discovery Rate (FDR) correction to p-values
    num_layers, num_units = p_values_matrix.shape
    adjusted_p_values = false_discovery_control(p_values_matrix.flatten())
    adjusted_p_values = adjusted_p_values.reshape((num_layers, num_units))

    # Save the computed mask and adjusted p-values
    np.save(save_path, language_mask)
    np.save(save_path_pvalues, adjusted_p_values)

    print(f"> {model_id} {network} mask cached to {save_path}")
    
    return language_mask
if __name__ == "__main__":
    """
    Entry point for localizing selective units in a language model. 
    This script allows the user to specify a model, configure localization settings, 
    and extract important model components for further processing.
    """

    # Argument parser to handle command-line inputs
    parser = argparse.ArgumentParser(description="Localize Units in LLMs")
    
    # Required argument: Hugging Face model name
    parser.add_argument("--model-name", type=str, required=True, help="Hugging Face model name")

    # Percentage of units to localize (optional, overrides num-units if provided)
    parser.add_argument("--percentage", type=float, default=None, 
                        help="Percentage of units to localize. Overrides num-units if provided.")

    # Range in percentiles for unit localization (e.g., "100-100" for top selective, "0-0" for least selective)
    parser.add_argument("--localize-range", type=str, default="100-100", 
                        help="Percentile range to localize units. Example: '100-100' selects top units, '0-0' selects least selective units.")

    # Network type: Selective unit localization for language, theory-of-mind, or multiple-demand networks
    parser.add_argument("--network", type=str, default="language", 
                        help="Network to localize. Options: 'language', 'theory-of-mind', 'multiple-demand'.")

    # Pooling strategy: Determines how token activations are aggregated
    parser.add_argument("--pooling", type=str, default="last-token", choices=["last-token", "mean"], 
                        help="Token aggregation method. Options: 'last-token' (default), 'mean'.")

    # Number of units to localize (optional, percentage takes priority if provided)
    parser.add_argument("--num-units", type=int, default=None, 
                        help="Number of units to localize. If 'percentage' is provided, it takes precedence.")

    # Random seed for reproducibility
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    # Device selection: CPU or GPU
    parser.add_argument("--device", type=str, default=None, 
                        help="Device to use for computation. Defaults to CUDA if available.")

    # Flag to use an untrained version of the model
    parser.add_argument("--untrained", action="store_true", 
                        help="Use an untrained version of the model (randomly initialized).")

    # Flag to overwrite cached localization masks
    parser.add_argument("--overwrite", action="store_true", 
                        help="Overwrite existing cached mask if it exists.")

    # Parse command-line arguments
    args = parser.parse_args()

    # Ensure that either a percentage or a specific number of units is provided
    assert args.percentage or args.num_units, "You must either provide a percentage of units to localize or a specific number of units."

    # Ensure the specified network is valid
    assert args.network in {"language", "theory-of-mind", "multiple-demand"}, "Unsupported network type. Choose from 'language', 'theory-of-mind', or 'multiple-demand'."

    # Assign parsed arguments to variables
    model_name = args.model_name
    pretrained = not args.untrained  # If --untrained is provided, use a randomly initialized model
    localize_range = args.localize_range
    num_units = args.num_units
    percentage = args.percentage
    pooling = args.pooling
    network = args.network
    seed = args.seed
    batch_size = 1  # Set batch size for processing

    # Determine device: Default to CUDA if available, otherwise use CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if args.device is None else args.device

    # Load the model: Either from a pretrained Hugging Face checkpoint or randomly initialized
    if pretrained:
        # model = transformers.AutoModelForCausalLM.from_pretrained(model_name) # for LLMs
        # load Llava in half precision
        model = LlavaForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
        processor = AutoProcessor.from_pretrained(model_name)
    else:
        model_config = transformers.AutoConfig.from_pretrained(model_name)
        model = transformers.AutoModelForCausalLM.from_config(config=model_config)

    # Load tokenizer corresponding to the model
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Ensure padding token is set

    # Extract the model name from the full path if applicable
    model_name = os.path.basename(model_name)

    # Retrieve model-specific layer names and hidden dimension size
    model_layer_names = get_layer_names(model_name)
    hidden_dim = get_hidden_dim(model_name)

    # Set model to evaluation mode (disables gradient computations)
    model.eval()

    # Call the localize function to perform selective unit localization
    localize(
        model_id=model_name,
        network=network,
        pooling=pooling,
        model=model,
        num_units=num_units,
        percentage=percentage,
        tokenizer=tokenizer,
        hidden_dim=hidden_dim,
        layer_names=model_layer_names,
        batch_size=batch_size,
        seed=seed,
        device=device,
        localize_range=localize_range,
        pretrained=pretrained,
        overwrite=args.overwrite,
    )
