import os
import json
import mlx.core as mx
import mlx.nn as nn

from mlx.utils import tree_flatten

def create_save_directory(base_path):
    os.makedirs(base_path, exist_ok=True)
    print(f"Using directory: {base_path}")
    return base_path

def print_trainable_parameters(model):
    """
    Copied from mlx-examples.
    """
    def nparams(m):
        if isinstance(m, (nn.QuantizedLinear, nn.QuantizedEmbedding)):
            return m.weight.size * (32 // m.bits)
        return sum(v.size for _, v in tree_flatten(m.parameters()))

    leaf_modules = tree_flatten(
        model.leaf_modules(), is_leaf=lambda m: isinstance(m, nn.Module)
    )
    total_p = sum(nparams(m) for _, m in leaf_modules) / 10**6
    print(f"Training model with: {total_p:.3f}M) Params.")

def save_model(model: nn.Module, save_path):
    model_path = os.path.join(save_path, "model.safetensors")
    flattened_tree = tree_flatten(model.trainable_parameters())
    mx.save_safetensors(model_path, dict(flattened_tree))
    print(f"Saved model to {str(save_path)}")

def save_config(args, save_path):
    config = vars(args)
    config_path = os.path.join(save_path, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Configuration file saved to {config_path}")