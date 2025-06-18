# Copyright © 2024-2025 Gökdeniz Gülmez

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import time

from kan.kan import KANLinear
from kan.architectures.mlp import GatedKANMLP, SmallKANMLP, MiddleKANMLP, BigKANMLP
from kan.architectures.conv import KANConv2d

class KANTester:
    """Automated test suite for KAN models"""
    
    def __init__(self):
        # Test data
        self.input_data = mx.array([[12, 4]])
        self.output_data = mx.array([[2, 4, 1]])
        self.input_data_conv = self.input_data.reshape(1, 1, 1, 2)
        self.output_data_conv = self.output_data.reshape(1, 1, 1, 3)
        
        # Model parameters
        self.in_features = 2
        self.hidden_dim = 4
        self.out_features = 3
        
        # Training parameters
        self.num_epochs = 100
        self.learning_rate = 0.0004
        self.weight_decay = 0.003
        
        # Results storage
        self.results = {}
        
    def test_mlp_models(self):
        """Test all MLP model variants"""
        mlp_models = {
            'SmallKANMLP': SmallKANMLP(self.in_features, self.hidden_dim, self.out_features),
            'MiddleKANMLP': MiddleKANMLP(self.in_features, self.hidden_dim, self.out_features),
            'BigKANMLP': BigKANMLP(self.in_features, self.hidden_dim, self.out_features),
            'GatedKANMLP': GatedKANMLP(self.in_features, self.hidden_dim, self.out_features)
        }
        
        for name, model in mlp_models.items():
            print(f"\n===== Testing {name} =====")
            print(model)
            
            # Forward pass test
            output = model(self.input_data)
            print(f"Output shape: {output.shape}")
            print(f"Output: {output}")
            
            # Training test
            self._train_model(name, model, self.input_data, self.output_data, is_conv=False)
    
    def test_conv_models(self):
        """Test convolutional model variants"""
        # Change kernel_size to match your input/output dimensions
        conv_model = KANConv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0)
        
        print("\n===== Testing KANConv2d =====")
        print(conv_model)
        
        # Forward pass test
        output = conv_model(self.input_data_conv)
        print(f"Output shape: {output.shape}")
        print(f"Output: {output}")
        
        # Adjust target data to match the output shape
        # Create a compatible target with the same shape as the output
        dummy_output = conv_model(self.input_data_conv)
        self.output_data_conv = mx.zeros_like(dummy_output)
        # Fill with some values
        self.output_data_conv = self.output_data_conv + 0.5
        
        # Training test
        self._train_model('KANConv2d', conv_model, self.input_data_conv, self.output_data_conv, is_conv=True)
    
    def _loss_fn_mlp(self, model, X, y):
        """Loss function for MLP models"""
        return mx.mean(nn.losses.cross_entropy(model(X), y))
    
    def _loss_fn_conv(self, model, X, y):
        """Loss function for convolutional models - using MSE instead of cross-entropy"""
        pred = model(X)
        # Use mean squared error which doesn't require specific shapes
        return mx.mean(mx.square(pred - y))
    
    def _train_model(self, name: str, model: nn.Module, X: mx.array, y: mx.array, is_conv: bool = False):
        """Train a model and record metrics"""
        print(f"Training {name}...")
        
        optimizer = optim.AdamW(learning_rate=self.learning_rate, weight_decay=self.weight_decay)
        
        # Select appropriate loss function based on model type
        loss_fn = self._loss_fn_conv if is_conv else self._loss_fn_mlp
        loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
        
        start_time = time.time()
        losses = []
        
        for epoch in range(self.num_epochs):
            # Training step
            loss, grads = loss_and_grad_fn(model, X, y)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            losses.append(loss.item())
            
            # Update grid points for KANLinear layers
            for layer_name, layer in model.__dict__.items():
                if isinstance(layer, KANLinear):
                    layer.update_grid(X)
            
            print(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {loss.item():.6f}")
        
        training_time = time.time() - start_time
        
        # Final inference test
        model.eval()
        inference_start = time.time()
        final_output = model(X)
        inference_time = time.time() - inference_start
        
        # Store results
        self.results[name] = {
            'initial_loss': losses[0],
            'final_loss': losses[-1],
            'training_time': training_time,
            'inference_time': inference_time,
            'loss_reduction': 1.0 - (losses[-1] / losses[0]) if losses[0] != 0 else 0
        }
        
        print(f"Training completed in {training_time:.4f}s")
        print(f"Loss reduced from {losses[0]:.6f} to {losses[-1]:.6f}")
    
    def run_all_tests(self):
        """Run all tests and print summary"""
        print("Starting KAN model tests...")
        
        # Run tests
        self.test_mlp_models()
        self.test_conv_models()
        
        # Print summary
        self._print_summary()
        
        return all(r['final_loss'] < r['initial_loss'] for r in self.results.values())
    
    def _print_summary(self):
        """Print test results summary"""
        print("\n" + "="*60)
        print("KAN MODELS TEST SUMMARY")
        print("="*60)
        
        # Table header
        print(f"{'Model':<15} {'Initial Loss':<15} {'Final Loss':<15} {'Reduction %':<15}")
        print("-"*60)
        
        # Table rows
        for name, result in self.results.items():
            reduction_pct = result['loss_reduction'] * 100
            print(f"{name:<15} {result['initial_loss']:<15.6f} {result['final_loss']:<15.6f} {reduction_pct:<15.2f}%")
            
        print("="*60)
        
        # Overall result
        all_improved = all(r['final_loss'] < r['initial_loss'] for r in self.results.values())
        print(f"Test {'PASSED' if all_improved else 'FAILED'}: {'All' if all_improved else 'Not all'} models showed loss reduction")
        print("="*60)


if __name__ == "__main__":
    # Run automated tests
    tester = KANTester()
    success = tester.run_all_tests()
    
    # Exit with appropriate code for CI/CD pipelines
    exit(0 if success else 1)