# Copyright © 2024-2025 Gökdeniz Gülmez

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.utils import load, save_weights, save_config
from mlx.utils import tree_flatten

from kan.kan import KANLinear

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class KANConversionConfig:
    """Configuration for KAN conversion."""
    grid_size: int = 5
    spline_order: int = 3
    scale_noise: float = 0.1
    scale_base: float = 1.0
    scale_spline: float = 1.0
    enable_standalone_scale_spline: bool = True
    hidden_act: str = 'silu'
    grid_eps: float = 0.02
    grid_range: List[float] = None
    bias: bool = True
    conversion_mode: str = 'mlp_only'  # 'mlp_only', 'all_linear', 'selective'
    preserve_embeddings: bool = True
    preserve_output_proj: bool = True
    preserve_attention: bool = True
    
    def __post_init__(self):
        if self.grid_range is None:
            self.grid_range = [-1.0, 1.0]

class MLXLMKANConverter:
    """Converter for MLX-LM models to use KAN layers."""
    
    def __init__(self, kan_config: Optional[KANConversionConfig] = None):
        self.kan_config = kan_config or KANConversionConfig()
        self.conversion_stats = {
            'total_layers_found': 0,
            'layers_converted': 0,
            'layers_skipped': 0,
            'original_params': 0,
            'new_params': 0,
            'converted_layer_names': []
        }
    
    def load_model(self, model_path: str) -> Tuple[nn.Module, Any]:
        """
        Load MLX-LM model and tokenizer.
        
        Args:
            model_path: HuggingFace repo or local path
            
        Returns:
            Tuple of (model, tokenizer)
        """
        logger.info(f"Loading model from {model_path}")
        model, tokenizer = load(model_path)
        logger.info(f"Model type: {type(model).__name__}")
        logger.info(f"Model config: {getattr(model, 'args', 'No args found')}")
        
        # Debug: Print model structure
        self._debug_model_structure(model)
        
        return model, tokenizer
    
    def _debug_model_structure(self, model: nn.Module, prefix: str = "", max_depth: int = 3):
        """Debug function to print model structure."""
        if max_depth <= 0:
            return
            
        logger.info(f"Model structure at {prefix or 'root'}:")
        
        # Get all attributes of the model
        for name in dir(model):
            if name.startswith('_'):
                continue
                
            try:
                attr = getattr(model, name)
                if isinstance(attr, nn.Module):
                    current_path = f"{prefix}.{name}" if prefix else name
                    logger.info(f"  {current_path}: {type(attr).__name__}")
                    
                    # Check if it's a Linear layer
                    if isinstance(attr, nn.Linear):
                        logger.info(f"    -> Linear layer: {attr.weight.shape}")
                    
                    # Recursively explore if it's not too deep
                    if max_depth > 1:
                        self._debug_model_structure(attr, current_path, max_depth - 1)
                        
                elif isinstance(attr, (list, tuple)):
                    for i, item in enumerate(attr):
                        if isinstance(item, nn.Module):
                            current_path = f"{prefix}.{name}[{i}]" if prefix else f"{name}[{i}]"
                            logger.info(f"  {current_path}: {type(item).__name__}")
                            
                            if isinstance(item, nn.Linear):
                                logger.info(f"    -> Linear layer: {item.weight.shape}")
                            
                            if max_depth > 1:
                                self._debug_model_structure(item, current_path, max_depth - 1)
                                
            except (AttributeError, TypeError, RuntimeError):
                continue
    
    def find_linear_layers_by_inspection(self, model: nn.Module, prefix: str = "") -> Dict[str, nn.Linear]:
        """
        Find linear layers by inspecting model attributes.
        
        Args:
            model: Module to search
            prefix: Current path prefix
            
        Returns:
            Dictionary mapping layer paths to Linear modules
        """
        linear_layers = {}
        
        # Check if current module is a Linear layer
        if isinstance(model, nn.Linear):
            linear_layers[prefix] = model
            return linear_layers
        
        # Inspect all attributes
        for name in dir(model):
            if name.startswith('_'):
                continue
                
            try:
                attr = getattr(model, name)
                current_path = f"{prefix}.{name}" if prefix else name
                
                if isinstance(attr, nn.Module):
                    # Recursively search this module
                    child_linear = self.find_linear_layers_by_inspection(attr, current_path)
                    linear_layers.update(child_linear)
                    
                elif isinstance(attr, (list, tuple)):
                    # Handle sequences of modules (like layers)
                    for i, item in enumerate(attr):
                        if isinstance(item, nn.Module):
                            item_path = f"{current_path}[{i}]"
                            item_linear = self.find_linear_layers_by_inspection(item, item_path)
                            linear_layers.update(item_linear)
                            
            except (AttributeError, TypeError, RuntimeError):
                continue
        
        return linear_layers
    
    def identify_mlp_layers(self, model: nn.Module) -> List[str]:
        """
        Identify MLP/feedforward layers in common transformer architectures.
        
        Args:
            model: MLX model
            
        Returns:
            List of layer paths for MLP components
        """
        mlp_layers = []
        
        # Get all linear layers
        all_linear = self.find_linear_layers_by_inspection(model)
        
        logger.info(f"All linear layers found: {list(all_linear.keys())}")
        
        for path, module in all_linear.items():
            # Common MLP layer patterns - updated for Qwen3 architecture
            path_lower = path.lower()
            if any(pattern in path_lower for pattern in [
                'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj',  # Llama/Mistral/Qwen3
                'mlp.fc1', 'mlp.fc2',  # GPT-style
                'mlp.c_fc', 'mlp.c_proj',  # GPT-2 style
                'feed_forward.w1', 'feed_forward.w2', 'feed_forward.w3',  # Some architectures
                'ffn.dense_h_to_4h', 'ffn.dense_4h_to_h',  # BLOOM style
                'intermediate.dense', 'output.dense',  # BERT style
                'mlp.linear', 'mlp.proj',  # Generic MLP patterns
                'mlp.dense'  # Another common pattern
            ]):
                mlp_layers.append(path)
        
        logger.info(f"Found {len(mlp_layers)} MLP layers: {mlp_layers}")
        return mlp_layers
    
    def identify_all_linear_layers(self, model: nn.Module) -> List[str]:
        """
        Identify all linear layers in the model.
        
        Args:
            model: MLX model
            
        Returns:
            List of all linear layer paths
        """
        all_linear = self.find_linear_layers_by_inspection(model)
        linear_layers = list(all_linear.keys())
        
        logger.info(f"Found {len(linear_layers)} linear layers")
        return linear_layers
    
    def should_convert_layer(self, layer_name: str) -> bool:
        """
        Determine if a layer should be converted based on configuration.
        
        Args:
            layer_name: Name of the layer
            
        Returns:
            True if layer should be converted
        """
        layer_lower = layer_name.lower()
        
        # Skip embeddings if configured
        if self.kan_config.preserve_embeddings:
            if any(pattern in layer_lower for pattern in [
                'embed', 'wte', 'wpe', 'token_embedding', 'position_embedding'
            ]):
                return False
        
        # Skip output projection if configured
        if self.kan_config.preserve_output_proj:
            if any(pattern in layer_lower for pattern in [
                'lm_head', 'output_proj', 'classifier', 'score'
            ]):
                return False
        
        # Skip attention layers if configured
        if self.kan_config.preserve_attention:
            if any(pattern in layer_lower for pattern in [
                'attn', 'attention', 'self_attn', 'q_proj', 'k_proj', 'v_proj', 'o_proj'
            ]):
                return False
        
        return True
    
    def get_layers_to_convert(self, model: nn.Module) -> List[str]:
        """
        Get list of layers to convert based on conversion mode.
        
        Args:
            model: MLX model
            
        Returns:
            List of layer names to convert
        """
        if self.kan_config.conversion_mode == 'mlp_only':
            layers = self.identify_mlp_layers(model)
        elif self.kan_config.conversion_mode == 'all_linear':
            layers = self.identify_all_linear_layers(model)
        else:  # selective
            layers = self.identify_all_linear_layers(model)
        
        # Apply filtering based on preservation settings
        filtered_layers = [layer for layer in layers if self.should_convert_layer(layer)]
        
        self.conversion_stats['total_layers_found'] = len(layers)
        logger.info(f"Will convert {len(filtered_layers)} out of {len(layers)} layers")
        
        return filtered_layers
    
    def create_kan_layer(self, original_layer: nn.Linear) -> KANLinear:
        """
        Create a KAN layer to replace a linear layer.
        
        Args:
            original_layer: Original linear layer
            
        Returns:
            KANLinear layer
        """
        in_features = original_layer.weight.shape[1]
        out_features = original_layer.weight.shape[0]
        has_bias = hasattr(original_layer, 'bias') and original_layer.bias is not None
        
        kan_layer = KANLinear(
            in_features=in_features,
            out_features=out_features,
            grid_size=self.kan_config.grid_size,
            spline_order=self.kan_config.spline_order,
            scale_noise=self.kan_config.scale_noise,
            scale_base=self.kan_config.scale_base,
            scale_spline=self.kan_config.scale_spline,
            enable_standalone_scale_spline=self.kan_config.enable_standalone_scale_spline,
            hidden_act=self.kan_config.hidden_act,
            grid_eps=self.kan_config.grid_eps,
            grid_range=self.kan_config.grid_range,
            bias=has_bias
        )
        
        # Initialize KAN layer with information from original layer
        self._initialize_kan_from_linear(kan_layer, original_layer)
        
        return kan_layer
    
    def _initialize_kan_from_linear(self, kan_layer: KANLinear, linear_layer: nn.Linear):
        """
        Initialize KAN layer using the original linear layer weights.
        
        Args:
            kan_layer: KAN layer to initialize
            linear_layer: Original linear layer
        """
        # Get weight statistics for scaling
        weight_std = mx.std(linear_layer.weight)
        weight_mean = mx.mean(linear_layer.weight)
        
        # Initialize base weight with scaled version of original
        kan_layer.base_weight = linear_layer.weight * self.kan_config.scale_base
        
        # Initialize spline weights with smaller magnitude
        spline_scale = weight_std * self.kan_config.scale_spline * 0.1
        kan_layer.spline_weight = kan_layer.spline_weight * spline_scale
        
        # Copy bias if it exists
        if hasattr(linear_layer, 'bias') and linear_layer.bias is not None and kan_layer.bias:
            kan_layer.bias_param = linear_layer.bias.copy()
        
        logger.debug(f"Initialized KAN layer with weight_std={weight_std:.4f}")
    
    def get_module_by_path(self, model: nn.Module, path: str) -> nn.Module:
        """
        Get a module by its path.
        
        Args:
            model: Root model
            path: Path to the module (e.g., "layers[0].mlp.gate_proj")
            
        Returns:
            The module at the specified path
        """
        if not path:
            return model
            
        current = model
        
        # Handle both dot notation and bracket notation
        path = path.replace('[', '.[').replace('].', '.')
        if path.endswith(']'):
            path = path[:-1] + '.'
        
        path_parts = path.split('.')
        path_parts = [p for p in path_parts if p]  # Remove empty parts
        
        for part in path_parts:
            if part.startswith('[') and part.endswith(']'):
                # Handle list/tuple indexing
                index = int(part[1:-1])
                current = current[index]
            else:
                current = getattr(current, part)
        
        return current
    
    def set_module_by_path(self, model: nn.Module, path: str, new_module: nn.Module):
        """
        Set a module at a specific path.
        
        Args:
            model: Root model
            path: Path to the module
            new_module: New module to set
        """
        if not path:
            raise ValueError("Cannot replace root model")
        
        # Handle both dot notation and bracket notation
        path = path.replace('[', '.[').replace('].', '.')
        if path.endswith(']'):
            path = path[:-1] + '.'
            
        path_parts = path.split('.')
        path_parts = [p for p in path_parts if p]  # Remove empty parts
        
        current = model
        
        # Navigate to parent
        for part in path_parts[:-1]:
            if part.startswith('[') and part.endswith(']'):
                index = int(part[1:-1])
                current = current[index]
            else:
                current = getattr(current, part)
        
        # Set the final attribute
        final_part = path_parts[-1]
        if final_part.startswith('[') and final_part.endswith(']'):
            index = int(final_part[1:-1])
            current[index] = new_module
        else:
            setattr(current, final_part, new_module)
    
    def replace_layer_in_model(self, model: nn.Module, layer_path: str, new_layer: nn.Module):
        """
        Replace a layer in the model with a new layer.
        
        Args:
            model: Model to modify
            layer_path: Path to the layer (e.g., "layers[0].mlp.gate_proj")
            new_layer: New layer to insert
        """
        try:
            # Get the old layer for statistics
            old_layer = self.get_module_by_path(model, layer_path)
            
            # Replace the layer
            self.set_module_by_path(model, layer_path, new_layer)
            
            # Update statistics
            if hasattr(old_layer, 'weight'):
                self.conversion_stats['original_params'] += old_layer.weight.size
                if hasattr(old_layer, 'bias') and old_layer.bias is not None:
                    self.conversion_stats['original_params'] += old_layer.bias.size
            
            # Count new parameters
            new_param_count = 0
            if hasattr(new_layer, 'base_weight'):
                new_param_count += new_layer.base_weight.size
            if hasattr(new_layer, 'spline_weight'):
                new_param_count += new_layer.spline_weight.size
            if hasattr(new_layer, 'spline_scaler'):
                new_param_count += new_layer.spline_scaler.size
            if hasattr(new_layer, 'bias_param') and new_layer.bias_param is not None:
                new_param_count += new_layer.bias_param.size
            if hasattr(new_layer, 'grid'):
                new_param_count += new_layer.grid.size
            
            self.conversion_stats['new_params'] += new_param_count
            
            logger.info(f"Replaced {layer_path} with KAN layer")
            
        except Exception as e:
            logger.error(f"Failed to replace layer {layer_path}: {e}")
            raise
    
    def update_model_args(self, model: nn.Module) -> nn.Module:
        """
        Update model args to include KAN configuration.
        
        Args:
            model: Model to update
            
        Returns:
            Updated model
        """
        if hasattr(model, 'args'):
            # Add KAN configuration to model args
            kan_args = asdict(self.kan_config)
            
            # Update existing args or create new ones
            if hasattr(model.args, '__dict__'):
                for key, value in kan_args.items():
                    setattr(model.args, key, value)
                
                # Add conversion metadata
                model.args.conversion_stats = self.conversion_stats
                model.args.kan_converted = True
                
                # Update model type if it exists
                if hasattr(model.args, 'model_type'):
                    model.args.model_type = f"{model.args.model_type}_kan"
            else:
                logger.warning("Model args is not a standard object, creating new args")
                
                # Create a simple args object
                class SimpleArgs:
                    def __init__(self, original_args, kan_config, conversion_stats):
                        # Copy original attributes if possible
                        if hasattr(original_args, '__dict__'):
                            for key, value in original_args.__dict__.items():
                                setattr(self, key, value)
                        
                        # Add KAN config
                        for key, value in kan_config.items():
                            setattr(self, key, value)
                        
                        # Add metadata
                        self.conversion_stats = conversion_stats
                        self.kan_converted = True
                        self.model_type = getattr(original_args, 'model_type', 'unknown') + '_kan'
                
                model.args = SimpleArgs(model.args, kan_args, self.conversion_stats)
            
            logger.info("Updated model args with KAN configuration")
        else:
            logger.warning("Model has no args attribute, cannot update configuration")
        
        return model
    
    def convert_model(self, model: nn.Module) -> nn.Module:
        """
        Convert model by replacing specified layers with KAN layers.
        
        Args:
            model: Original model
            
        Returns:
            Converted model with KAN layers
        """
        logger.info("Starting model conversion to KAN")
        
        # Get layers to convert
        layers_to_convert = self.get_layers_to_convert(model)
        
        if not layers_to_convert:
            logger.warning("No layers found to convert")
            return model
        
        # Convert each layer
        for layer_path in layers_to_convert:
            try:
                # Get the original layer
                original_layer = self.get_module_by_path(model, layer_path)
                
                if not isinstance(original_layer, nn.Linear):
                    logger.warning(f"Layer {layer_path} is not a Linear layer, skipping")
                    self.conversion_stats['layers_skipped'] += 1
                    continue
                
                # Create KAN replacement
                kan_layer = self.create_kan_layer(original_layer)
                
                # Replace in model
                self.replace_layer_in_model(model, layer_path, kan_layer)
                
                self.conversion_stats['layers_converted'] += 1
                self.conversion_stats['converted_layer_names'].append(layer_path)
                
            except Exception as e:
                logger.error(f"Failed to convert layer {layer_path}: {e}")
                self.conversion_stats['layers_skipped'] += 1
        
        # Update model args
        model = self.update_model_args(model)
        
        logger.info(f"Conversion complete: {self.conversion_stats}")
        return model
    
    def save_converted_model(self, model: nn.Module, tokenizer: Any, save_path: str):
        """
        Save the converted model using mlx_lm utilities.
        
        Args:
            model: Converted model
            tokenizer: Tokenizer
            save_path: Path to save the model
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving converted model to {save_path}")
        
        # Save model weights - Use a simpler approach for MLX models
        try:
            # Try to get parameters using MLX's approach
            weights = {}
            
            # Method 1: Try to use model.parameters() if available
            if hasattr(model, 'parameters') and callable(model.parameters):
                try:
                    params = model.parameters()
                    if isinstance(params, dict):
                        weights = params
                    else:
                        # If parameters() returns something else, try to convert it
                        weights = dict(params) if hasattr(params, 'items') else {}
                except:
                    pass
            
            # Method 2: Fallback to manual parameter collection
            if not weights:
                def collect_params(module, prefix=""):
                    for name in dir(module):
                        if name.startswith('_'):
                            continue
                        try:
                            attr = getattr(module, name)
                            if isinstance(attr, mx.array):
                                param_name = f"{prefix}.{name}" if prefix else name
                                weights[param_name] = attr
                            elif isinstance(attr, nn.Module):
                                new_prefix = f"{prefix}.{name}" if prefix else name
                                collect_params(attr, new_prefix)
                            elif isinstance(attr, (list, tuple)):
                                for i, item in enumerate(attr):
                                    if isinstance(item, nn.Module):
                                        new_prefix = f"{prefix}.{name}[{i}]" if prefix else f"{name}[{i}]"
                                        collect_params(item, new_prefix)
                        except:
                            continue
                
                collect_params(model)
            
            if weights:
                save_weights(str(save_path), weights)
                logger.info(f"Saved {len(weights)} parameters")
            else:
                logger.warning("No parameters found to save")
            
        except Exception as e:
            logger.error(f"Error saving weights: {e}")
            raise
        
        # Save model configuration
        config_dict = {}
        if hasattr(model, 'args'):
            if hasattr(model.args, '__dict__'):
                config_dict = {k: v for k, v in model.args.__dict__.items() 
                              if not k.startswith('_') and isinstance(v, (str, int, float, bool, list, dict, type(None)))}
            else:
                config_dict = {'model_type': str(type(model.args))}
        
        save_config(config_dict, str(save_path))
        
        # Save tokenizer
        if hasattr(tokenizer, 'save_pretrained'):
            try:
                tokenizer.save_pretrained(str(save_path))
            except Exception as e:
                logger.warning(f"Could not save tokenizer: {e}")
        
        # Save conversion metadata
        metadata = {
            'conversion_config': asdict(self.kan_config),
            'conversion_stats': self.conversion_stats,
            'mlx_lm_kan_version': '1.0.0'
        }
        
        with open(save_path / 'kan_conversion_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved successfully to {save_path}")

def convert_mlx_lm_to_kan(
    model_path: str,
    save_path: str,
    kan_config: Optional[KANConversionConfig] = None,
    conversion_mode: str = 'mlp_only'
) -> Dict[str, Any]:
    """
    Main function to convert MLX-LM model to use KAN layers.
    
    Args:
        model_path: HuggingFace repo or local path to original model
        save_path: Path to save converted model
        kan_config: KAN conversion configuration
        conversion_mode: 'mlp_only', 'all_linear', or 'selective'
        
    Returns:
        Conversion statistics
    """
    # Set up configuration
    if kan_config is None:
        kan_config = KANConversionConfig()
    kan_config.conversion_mode = conversion_mode
    
    # Initialize converter
    converter = MLXLMKANConverter(kan_config)
    
    try:
        # Load model
        model, tokenizer = converter.load_model(model_path)
        
        # Convert model
        converted_model = converter.convert_model(model)
        
        # Save converted model
        converter.save_converted_model(converted_model, tokenizer, save_path)
        
        return converter.conversion_stats
        
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        raise

# Convenience functions for different conversion modes
def convert_mlp_to_kan(model_path: str, save_path: str, **kan_kwargs) -> Dict[str, Any]:
    """Convert only MLP layers to KAN."""
    kan_config = KANConversionConfig(**kan_kwargs)
    return convert_mlx_lm_to_kan(model_path, save_path, kan_config, 'mlp_only')

def convert_all_linear_to_kan(model_path: str, save_path: str, **kan_kwargs) -> Dict[str, Any]:
    """Convert all linear layers to KAN."""
    kan_config = KANConversionConfig(**kan_kwargs)
    return convert_mlx_lm_to_kan(model_path, save_path, kan_config, 'all_linear')

def convert_selective_to_kan(model_path: str, save_path: str, **kan_kwargs) -> Dict[str, Any]:
    """Convert linear layers selectively (preserving embeddings, attention, etc.)."""
    kan_config = KANConversionConfig(**kan_kwargs)
    return convert_mlx_lm_to_kan(model_path, save_path, kan_config, 'selective')

# Example usage
if __name__ == "__main__":
    # Example 1: Convert only MLP layers
    stats = convert_mlp_to_kan(
        model_path="mlx-community/Josiefied-Qwen3-0.6B-abliterated-v1-4bit",
        save_path="/Users/gokdenizgulmez/Desktop/mlx-kan/Josiefied_KAN",
        grid_size=8,
        spline_order=3,
        hidden_act='gelu'
    )
    print(f"MLP conversion stats: {stats}")