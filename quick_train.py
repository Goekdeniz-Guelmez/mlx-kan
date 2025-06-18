# Copyright © 2024-2025 Gökdeniz Gülmez

import argparse, logging, time, gzip, os, pickle
import mlx.optimizers as optim
from urllib import request
from pathlib import Path
from tqdm import tqdm

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from kan.architectures.mlp import GatedKANMLP
from global_utils.utils import get_parameters, save_model, save_config, create_save_directory


def mnist(
    save_dir="/tmp", base_url="https://raw.githubusercontent.com/fgnt/mnist/master/", filename="mnist.pkl"
):
    """
    Load the MNIST dataset in 4 tensors: train images, train labels,
    test images, and test labels.

    Checks `save_dir` for already downloaded data otherwise downloads.

    Download code modified from:
      https://github.com/hsjeong5/MNIST-for-Numpy
    """

    def download_and_save(save_file):
        filename = [
            ["training_images", "train-images-idx3-ubyte.gz"],
            ["test_images", "t10k-images-idx3-ubyte.gz"],
            ["training_labels", "train-labels-idx1-ubyte.gz"],
            ["test_labels", "t10k-labels-idx1-ubyte.gz"],
        ]

        mnist = {}
        for name in filename:
            out_file = os.path.join("/tmp", name[1])
            request.urlretrieve(base_url + name[1], out_file)
        for name in filename[:2]:
            out_file = os.path.join("/tmp", name[1])
            with gzip.open(out_file, "rb") as f:
                mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(
                    -1, 28 * 28
                )
        for name in filename[-2:]:
            out_file = os.path.join("/tmp", name[1])
            with gzip.open(out_file, "rb") as f:
                mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
        with open(save_file, "wb") as f:
            pickle.dump(mnist, f)

    save_file = os.path.join(save_dir, filename)
    if not os.path.exists(save_file):
        download_and_save(save_file)
    with open(save_file, "rb") as f:
        mnist = pickle.load(f)

    def preproc(x):
        return x.astype(np.float32) / 255.0

    mnist["training_images"] = preproc(mnist["training_images"])
    mnist["test_images"] = preproc(mnist["test_images"])
    return (
        mnist["training_images"],
        mnist["training_labels"].astype(np.uint32),
        mnist["test_images"],
        mnist["test_labels"].astype(np.uint32),
    )


def fashion_mnist(save_dir="/tmp"):
    return mnist(
        save_dir,
        base_url="http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",
        filename="fashion_mnist.pkl",
    )


def setup_logging(save_path):
    """Setup logging configuration."""
    log_dir = Path(save_path)
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_dir / "training.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

class Trainer:
    def __init__(self, args):
        self.args = args
        self.setup_seed(args.seed)
        self.setup_device(args.cpu)
        self.load_dataset(args.dataset)
        self.create_model(args)
        self.setup_optimizer(args.learning_rate, args.weight_decay)
        self.loss_and_grad_fn = nn.value_and_grad(self.model, self.loss_fn)
        
    def setup_seed(self, seed):
        """Set random seed for reproducibility."""
        np.random.seed(seed)
        
    def setup_device(self, use_cpu):
        """Set the computation device."""
        if use_cpu:
            mx.set_default_device(mx.cpu)
            
    def load_dataset(self, dataset_name):
        """Load and prepare the dataset."""
        print("\nLoading Dataset...")
        
        # Use globals() to access the function by name
        if dataset_name == "mnist":
            data_function = mnist
        elif dataset_name == "fashion_mnist":
            data_function = fashion_mnist
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
            
        self.train_images, self.train_labels, self.test_images, self.test_labels = map(
            mx.array, data_function()
        )
        print(f"Dataset '{dataset_name}' loaded successfully")
        
    def create_model(self, args):
        """Initialize the model."""
        print("\nInitializing model...")
        
        # For MNIST/Fashion-MNIST, input is 784 (flattened 28x28 images)
        # and output should be 10 (number of classes)
        input_dim = 784  # Flattened 28x28 images
        output_dim = 10  # Number of classes (digits 0-9 or fashion items)
        
        self.model = GatedKANMLP(
            in_features=input_dim,
            hidden_dim=args.hidden_dim,
            out_features=output_dim
        )
        
        mx.eval(self.model.parameters())
        total_params = get_parameters(self.model)
        print(f"Model created with {total_params:,} parameters")
        
    def setup_optimizer(self, learning_rate, weight_decay):
        """Setup the optimizer."""
        self.optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
        
    def loss_fn(self, model, X, y):
        """Calculate cross entropy loss."""
        return mx.mean(nn.losses.cross_entropy(model(X), y))
    
    def batch_iterate(self, batch_size, X, y):
        """Create batches for training/evaluation."""
        perm = mx.array(np.random.permutation(y.size))
        for s in range(0, y.size, batch_size):
            ids = perm[s : s + batch_size]
            yield X[ids], y[ids]
    
    def train_epoch(self, batch_size, clip_grad_norm=False):
        """Train for one epoch."""
        total_loss = 0
        num_batches = 0
        
        self.model.train()
        for X, y in tqdm(self.batch_iterate(batch_size, self.train_images, self.train_labels), 
                         total=len(self.train_images) // batch_size, 
                         desc='Training'):
            loss, grads = self.loss_and_grad_fn(self.model, X, y)
            
            if mx.isnan(loss).any():
                raise ValueError("Encountered NaN in loss")
                
            if clip_grad_norm:
                optim.clip_grad_norm(grads, max_norm=1.0)
                
            self.optimizer.update(self.model, grads)
            mx.eval(self.model.parameters(), self.optimizer.state)
            
            total_loss += loss.item()
            num_batches += 1
            
        return total_loss / num_batches
    
    def evaluate(self, X, y, batch_size):
        """Evaluate the model."""
        self.model.eval()
        total_correct = 0
        total_samples = 0
        
        for batch_X, batch_y in self.batch_iterate(batch_size, X, y):
            predictions = mx.argmax(self.model(batch_X), axis=1)
            total_correct += mx.sum(predictions == batch_y)
            total_samples += batch_y.size
            
        return total_correct / total_samples
    
    def train(self):
        """Train the model for specified number of epochs."""
        args = self.args
        print(f"\nStarting training for {args.num_epochs} epochs")
        
        for e in range(args.num_epochs):
            current_epoch = e + 1
            
            # Train for one epoch
            tic = time.perf_counter()
            loss = self.train_epoch(args.batch_size, clip_grad_norm=args.clip_grad_norm)
            toc = time.perf_counter()
            
            log_msg = f"Epoch {current_epoch}/{args.num_epochs}: Train Loss: {loss:.4f}, Time: {toc - tic:.2f}s"
            logging.info(log_msg)
            print(log_msg)
            
            # Evaluate if needed
            if current_epoch == 1 or current_epoch % args.eval_interval == 0 or current_epoch == args.num_epochs:
                tic = time.perf_counter()
                accuracy = self.evaluate(self.test_images, self.test_labels, args.batch_size)
                toc = time.perf_counter()
                
                eval_msg = f"Epoch {current_epoch}/{args.num_epochs}: Test accuracy: {accuracy.item():.4f}, Time: {toc - tic:.2f}s"
                logging.info(eval_msg)
                print(eval_msg)
        
        # Final evaluation
        mx.eval(self.model.parameters(), self.optimizer.state)
        
    def save(self):
        """Save the trained model and configuration."""
        save_dir = create_save_directory(self.args.save_path)
        save_model(model=self.model, save_path=save_dir)
        save_config(self.args, save_dir)
        print(f"Model and configuration saved to {save_dir}")

def main():
    parser = argparse.ArgumentParser(description="Train KAN on MNIST with MLX")
    
    # Device settings
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of Metal GPU backend")
    
    # Dataset settings
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "fashion_mnist"], 
                        help="Dataset to use")
    
    # Model architecture
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension size")
    parser.add_argument("--in-features", type=int, default=28, help="Input feature dimension")
    parser.add_argument("--out-features", type=int, default=28, help="Output feature dimension")
    
    # Training settings
    parser.add_argument("--num-epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--clip-grad-norm", action="store_true", help="Apply gradient clipping")
    parser.add_argument("--eval-interval", type=int, default=1, help="Evaluate every N epochs")
    
    # Misc settings
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--save-path", type=str, default="trained_models/kan", 
                        help="Directory to save model and logs")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.save_path)
    
    # Create and train the model
    trainer = Trainer(args)
    trainer.train()
    trainer.save()

if __name__ == "__main__":
    main()

