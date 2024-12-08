import torch
from collections import OrderedDict
import numpy as np
import os
import random

WORD_MASK = "<mask>"

def sampler_func(clip, sn, random_choice=True):
    if random_choice:
        f = lambda n: [(lambda n, arr: n if arr == [] else np.random.choice(arr))(n * i / sn,
                                                                                range(int(n * i / sn),
                                                                                        max(int(n * i / sn) + 1,
                                                                                            int(n * (
                                                                                                    i + 1) / sn))))
                        for i in range(sn)]
    else:
        f = lambda n: [(lambda n, arr: n if arr == [] else int(np.mean(arr)))(n * i / sn, range(int(n * i / sn),
                                                                                                max(int(
                                                                                                    n * i / sn) + 1,
                                                                                                    int(n * (
                                                                                                            i + 1) / sn))))
                        for i in range(sn)]
    return f(clip)


def NoiseInjecting(raw_gloss, noise_rate=0.15, noise_type='omit_last', random_shuffle=False, is_train=True):
    new_gloss = []
    noise_idxes = []
    for ii, gloss in enumerate(raw_gloss):
        noise_num = -1 
        text = gloss.split()

        if noise_type == 'omit':
            # del noise
            if random.uniform(0, 1) <= 1. and is_train:
                index = sampler_func(len(text), int(len(text)*(1. - noise_rate)), random_choice=is_train)
                noise_gloss = []
                noise_idx = []
                for i, d in enumerate(text):
                    if i in index:
                        noise_gloss.append(d)
                    else:
                        noise_gloss.append(WORD_MASK)
                        noise_idx.append(i)
            else:
                noise_gloss = [d for d in text]

        elif noise_type == 'omit_last' :
            noise_idx = []
            if random.uniform(0, 1) <= 1.0 and is_train:
                index = np.arange(0, len(text) - int(np.ceil(len(text)*(np.random.uniform(0,noise_rate,(1,))))), 1, dtype=int)
                noise_gloss = []
                for i, d in enumerate(text):
                    if i in index:
                        noise_gloss.append(d)
                    else:
                        noise_gloss.append(WORD_MASK)
                        noise_idx.append(noise_num - 2)
                        noise_num -= 1
            else:
                noise_gloss = [d for d in text]
        
        if is_train and random_shuffle and random.uniform(0, 1) > 0.5:
            random.shuffle(noise_gloss) # random shuffle sequence

        new_gloss.append(' '.join(noise_gloss))
        noise_idxes.append(noise_idx)
    return new_gloss, noise_idxes

def manage_directory(path):
    # Check if the directory exists
    if not os.path.exists(path):
        # Create the directory if it doesn't exist
        os.makedirs(path)
        print(f"Directory '{path}' created.")
    else:
        # Remove all .csv files in the directory
        for filename in os.listdir(path):
            if filename.endswith(".csv"):  # Only target .csv files
                file_path = os.path.join(path, filename)
                try:
                    os.unlink(file_path)  # Remove the .csv file
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")
        print(f"All .csv files in the directory '{path}' have been removed.")


def extract_layers_by_prefix(checkpoint_path, prefixes):
    """
    Extract layers that start with specified prefixes from checkpoint.
    
    Args:
        checkpoint_path (str): Path to the checkpoint file
        prefixes (list): List of prefixes to match (e.g., ['feature_1', 'feature_2'])
    Returns:
        dict: Dictionary containing matched layer names and their shapes
        dict: Filtered state dictionary containing only matched layers
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Get state dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
        
    # Find matching layers
    matched_layers = {}
    filtered_state_dict = OrderedDict()
    
    for layer_name, tensor in state_dict.items():
        # Check if layer name starts with any of the prefixes
        if any(layer_name.startswith(prefix) for prefix in prefixes):
            matched_layers[layer_name] = tensor.shape
            filtered_state_dict[layer_name] = tensor
            
    # # Print matched layers
    # print("\n=== Matched Layers ===")
    # for name, shape in matched_layers.items():
    #     print(f"Layer: {name}")
    #     print(f"Shape: {shape}")
    #     print("-" * 50)
    # exit(0)
        
    return matched_layers, filtered_state_dict
import torch.nn.functional as F
class KLLoss(torch.nn.Module):
    """Loss that uses a 'hinge' on the lower bound.
    This means that for samples with a label value smaller than the threshold, the loss is zero if the prediction is
    also smaller than that threshold.
    args:
        error_matric:  What base loss to use (MSE by default).
        threshold:  Threshold to use for the hinge.
        clip:  Clip the loss if it is above this value.
    """

    def __init__(self, error_metric=torch.nn.KLDivLoss(size_average=True, reduce=True)):
        super().__init__()
        # print('=========using KL Loss=and has temperature and * bz==========')
        self.error_metric = error_metric

    def forward(self, prediction, label):
        batch_size = prediction.shape[0]
        probs1 = F.log_softmax(prediction, 1)
        probs2 = F.softmax(label * 10, 1)
        loss = self.error_metric(probs1, probs2) * batch_size
        return loss