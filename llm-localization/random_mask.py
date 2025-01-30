import os
import numpy as np
from utils import get_layer_names, get_hidden_dim

BRAINIO_CACHE = os.environ.get("BRAINIO", f"/mnt/u14157_ic_nlp_001_files_nfs/nlpdata1/home/bkhmsi/.brainio")

if  __name__ == "__main__":

    percentage = 1
    num_units = None
    pooling = "last-token"
    network = "lang"
    pretrained = False

    for model_name in [
        # "microsoft/Phi-3.5-mini-instruct",

        # "mistralai/Mistral-7B-v0.3",
        # "mistralai/Mistral-7B-Instruct-v0.3",

        # "meta-llama/Llama-2-7b-hf",
        # "meta-llama/Llama-2-13b-hf",
        # "meta-llama/Llama-2-7b-chat-hf",
        # "meta-llama/Llama-2-13b-chat-hf",
        # "meta-llama/Meta-Llama-3.1-8B-Instruct",

        # "google/gemma-2b",
        # "google/gemma-2b-it",
        # "google/gemma-7b",
        # "google/gemma-1.1-7b-it",

        # "tiiuae/falcon-7b",
        # "tiiuae/falcon-7b-instruct",
        # "gpt2-xl",
        "gpt2-large",
    ]:
        print(f"> Model: {model_name}")
        model_name = os.path.basename(model_name)
        language_mask = np.load(f"{BRAINIO_CACHE}/{model_name}_network={network}_pooling={pooling}_range=100-100_perc={percentage}_pretrained={pretrained}_abs.npy")
        print(f"Language mask shape: {language_mask.shape} | Num units: {np.sum(language_mask)}")
        for seed in [42, 43, 44]:

            layer_names = get_layer_names(model_name)
            hidden_dim = get_hidden_dim(model_name)

            if percentage is not None:
                num_units = int((percentage/100) * hidden_dim*len(layer_names))
                print(f"> Percentage: {percentage}% --> # Units: {num_units}")

            total_num_units = hidden_dim*len(layer_names)

            invlang_mask_indices = np.arange(total_num_units)[(1 - language_mask).flatten().astype(bool)]
            np.random.seed(seed)
            rand_indices = np.random.choice(invlang_mask_indices, size=num_units, replace=False)
            lang_mask_rand = np.full(total_num_units, 0)
            lang_mask_rand[rand_indices] = 1
            assert np.sum(lang_mask_rand) == num_units
            random_mask = lang_mask_rand.reshape((len(layer_names), hidden_dim))

            print(f"> Random mask shape: {random_mask.shape} | Num units: {np.sum(random_mask)}")
            save_path = f"{BRAINIO_CACHE}/{model_name}_network={network}_random={seed}_perc={percentage}_pretrained={pretrained}.npy"
            np.save(save_path, random_mask)

        