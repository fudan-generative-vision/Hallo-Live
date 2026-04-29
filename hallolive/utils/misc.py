import numpy as np
import random
import torch
import torch.nn as nn
from safetensors.torch import load_file
import torch.distributed as dist


def set_seed(seed: int, deterministic: bool = False):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`.

    Args:
        seed (`int`):
            The seed to set.
        deterministic (`bool`, *optional*, defaults to `False`):
            Whether to use deterministic algorithms where available. Can slow down training.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)


def merge_dict_list(dict_list):
    if len(dict_list) == 1:
        return dict_list[0]

    merged_dict = {}
    for k, v in dict_list[0].items():
        if isinstance(v, torch.Tensor):
            if v.ndim == 0:
                merged_dict[k] = torch.stack([d[k] for d in dict_list], dim=0)
            else:
                merged_dict[k] = torch.cat([d[k] for d in dict_list], dim=0)
        else:
            # for non-tensor values, we just copy the value from the first item
            merged_dict[k] = v
    return merged_dict


def load_ckpt(ckpt_path: str = ""):
    if ckpt_path.endswith(".safetensors"):
        state_dict = load_file(ckpt_path, device="cpu")
    elif ckpt_path.endswith(".pt"):
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True, mmap=True)
    else:
        raise RuntimeError("Only support .safetensors and .pt checkpoints currently")
    return state_dict


def count_params(model: nn.Module):
    num_params = sum(p.numel() for p in model.parameters())
    return num_params


def safe_load_state_dict(model, state_dict):
    """
    Loads state_dict into model, automatically filtering out keys that
    don't exist or have mismatched shapes.
    """
    # 1. Get the current model's parameter dictionary
    model_dict = model.state_dict()

    # 2. Filter out non-existent keys and shape mismatches
    filtered_dict = {}
    mismatched_keys = []

    for k, v in state_dict.items():
        if k in model_dict:
            if v.shape == model_dict[k].shape:
                filtered_dict[k] = v
            else:
                # Store info about mismatched shapes for debugging
                mismatched_keys.append((k, v.shape, model_dict[k].shape))

    # 4. Load the filtered weights
    # strict=False is essential since we deliberately removed some keys
    model.load_state_dict(filtered_dict, strict=False)

    if dist.get_rank() == 0:
        print(f"Successfully loaded {len(filtered_dict)} parameters.")
        print(f"Skipped {len(mismatched_keys)} mismatched parameters.")


def normalize_scalar_across_ranks(value, device: torch.device, eps: float = 1e-6):
    value_tensor = torch.as_tensor(value, device=device, dtype=torch.float32)

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        gathered_values = [torch.empty_like(value_tensor) for _ in range(world_size)]
        torch.distributed.all_gather(gathered_values, value_tensor)
        gathered_values = torch.stack(gathered_values)
    else:
        gathered_values = value_tensor.unsqueeze(0)

    global_mean = gathered_values.mean()
    global_var = gathered_values.var(unbiased=False)
    global_std = torch.sqrt(global_var)
    normalized_value = (value_tensor - global_mean) / global_std.clamp_min(eps)

    return value_tensor, normalized_value
