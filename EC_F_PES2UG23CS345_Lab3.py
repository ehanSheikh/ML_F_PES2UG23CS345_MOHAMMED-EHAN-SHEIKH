# lab.py
import torch

def get_entropy_of_dataset(tensor: torch.Tensor):

    target_col = tensor[:, -1]
    values, counts = torch.unique(target_col, return_counts=True)
    probs = counts.float() / counts.sum()
    entropy = -(probs * torch.log2(probs)).sum().item()
    return entropy


def get_avg_info_of_attribute(tensor: torch.Tensor, attribute: int):

    total_len = len(tensor)
    attr_col = tensor[:, attribute]
    values = torch.unique(attr_col)

    avg_info = 0.0
    for v in values:
        subset = tensor[attr_col == v]
        weight = len(subset) / total_len
        subset_entropy = get_entropy_of_dataset(subset)
        avg_info += weight * subset_entropy

    return avg_info


def get_information_gain(tensor: torch.Tensor, attribute: int):

    dataset_entropy = get_entropy_of_dataset(tensor)
    avg_info = get_avg_info_of_attribute(tensor, attribute)
    ig = dataset_entropy - avg_info
    return round(ig, 4)


def get_selected_attribute(tensor: torch.Tensor):

    num_attributes = tensor.shape[1] - 1
    ig_dict = {}
    for attr in range(num_attributes):
        ig_dict[attr] = get_information_gain(tensor, attr)
    best_attr = max(ig_dict, key=ig_dict.get)
    return ig_dict, best_attr
