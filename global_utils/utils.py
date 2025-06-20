# Copyright © 2024-2025 Gökdeniz Gülmez

from typing import Any, Union, Callable
import os
import json
import mlx.core as mx
import mlx.nn as nn

from mlx.utils import tree_flatten, tree_unflatten

from kan.args import ModelArgs
from kan.kan import KAN

def create_save_directory(base_path):
    os.makedirs(base_path, exist_ok=True)
    print(f"Using directory: {base_path}")
    return base_path

def get_parameters(model):
    def nparams(m):
        if isinstance(m, (nn.QuantizedLinear, nn.QuantizedEmbedding)):
            return m.weight.size * (32 // m.bits)
        return sum(v.size for _, v in tree_flatten(m.parameters()))

    leaf_modules = tree_flatten(
        model.leaf_modules(), is_leaf=lambda m: isinstance(m, nn.Module)
    )
    total_p = sum(nparams(m) for _, m in leaf_modules) / 10**6
    return total_p

def get_num_parameters(model) -> int:
    total_params = 0
    for layer in model.layers:
        # Count base weights
        total_params += layer.base_weight.size
        # Count spline weights
        total_params += layer.spline_weight.size
        # Count spline scalers if enabled
        if layer.config.enable_standalone_scale_spline:
            total_params += layer.spline_scaler.size
        # Count bias if enabled
        if layer.config.bias:
            total_params += layer.bias_param.size
    
    return total_params

def save_model(model: nn.Module, save_path):
    model_path = os.path.join(save_path, "model.safetensors")
    flattened_tree = tree_flatten(model.trainable_parameters())
    mx.save_safetensors(model_path, dict(flattened_tree))
    print(f"Saved model to {str(save_path)}")

def save_config(model_args: Any, save_path: str, train_args: Any = None):
    def is_json_serializable(value):
        try:
            json.dumps(value)
            return True
        except (TypeError, OverflowError):
            return False

    def serialize_dataclass(dataclass_instance):
        if not hasattr(dataclass_instance, "__dict__"):
            return dataclass_instance
        return {k: v if is_json_serializable(v) else str(v) for k, v in dataclass_instance.__dict__.items() if not k.startswith("__") and not callable(v)}

    config = serialize_dataclass(model_args)
    if train_args is not None:
        config.update(serialize_dataclass(train_args))
    
    config_path = os.path.join(save_path, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Configuration file saved to {config_path}")


def load_config(file_path: str) -> ModelArgs:
    with open(file_path, 'r') as f:
        config_dict = json.load(f)
    
    valid_keys = set(ModelArgs.__dataclass_fields__.keys())
    config_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
    
    return ModelArgs(**config_dict)


def load_model(folder_path: str):
    config_path = os.path.join(folder_path, "config.json")
    model_path = os.path.join(folder_path, "model.safetensors")
    
    config = load_config(config_path)
    model = KAN(config)
    
    weights = tree_unflatten(list(mx.load(model_path).items()))
    model.update(weights)
    
    return model