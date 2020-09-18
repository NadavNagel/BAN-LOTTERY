import numpy as np
import torch
def make_mask(model):
    step = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            step = step + 1
    mask = [None]* step
    step = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            tensor = param.data.cpu().numpy()
            mask[step] = np.ones_like(tensor)
            step = step + 1
    step = 0
    return mask, step


def prune_by_percentile(percent, mask, model, old_mask=None, prune_net=True, resample=False, reinit=False, **kwargs):

    # Calculate percentile value
    step = 0
    for name, param in model.named_parameters():

        # We do not prune bias term
        if 'weight' in name:
            tensor = param.data.cpu().numpy()
            alive = tensor[np.nonzero(tensor)]  # flattened array of nonzero values
            # percentile_value = np.percentile(abs(alive), percent)
            percentile_value = np.percentile(abs(tensor), percent)

            # Convert Tensors to numpy and calculate
            weight_dev = param.device
            if old_mask is None:
                new_mask = np.where(abs(tensor) < percentile_value, 0, mask[step])
            else:
                new_mask = old_mask[step]

            # Apply new weight and mask
            if prune_net:
                param.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
            mask[step] = new_mask
            step += 1

    step = 0
    return mask, step
