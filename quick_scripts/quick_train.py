import argparse
import time
import logging
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import quick_scripts.mnist as mnist

from kan import KAN

from quick_scripts.utils import print_trainable_parameters, save_model, save_config, create_save_directory

def create_model(args) -> nn.Module:
    print("Initializing and creating model")
    if args.use_kan_convolution:
        raise NotImplementedError("KAN Convolutional Layer functionality is not yet implemented.")
    else:
        layers = [args.in_features * args.out_features] + [args.hidden_dim] * (args.num_layers - 1) + [args.num_classes]
        return KAN(layers)

def loss_fn(model, X, y):
    return mx.mean(nn.losses.cross_entropy(model(X), y))

def eval_fn(model, X, y):
    return mx.mean(mx.argmax(model(X), axis=1) == y)

def batch_iterate(batch_size, X, y):
    perm = mx.array(np.random.permutation(y.size))
    for s in range(0, y.size, batch_size):
        ids = perm[s : s + batch_size]
        yield X[ids], y[ids]

def batched_eval_fn(model: nn.Module, X: mx.array, y: mx.array, batch_size: int) -> float:
    total_correct = 0
    total_samples = 0
    for batch_X, batch_y in batch_iterate(batch_size, X, y):
        predictions = mx.argmax(model(batch_X), axis=1)
        total_correct += mx.sum(predictions == batch_y)
        total_samples += batch_y.size
    return total_correct / total_samples

def train_epoch_batched(model: nn.Module, optimizer: optim.Optimizer, train_images: mx.array, train_labels: mx.array, batch_size: int, loss_and_grad_fn) -> float:
    model.train()
    total_loss = 0
    num_batches = 0
    for X, y in batch_iterate(batch_size, train_images, train_labels):
        loss, grads = loss_and_grad_fn(model, X, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        total_loss += loss.item()
        num_batches += 1
    return total_loss / num_batches

def train_epoch(model: nn.Module, optimizer: optim.Optimizer, train_images: mx.array, train_labels: mx.array, batch_size: int, loss_and_grad_fn) -> float:
    model.train()
    for X, y in batch_iterate(batch_size, train_images, train_labels):
        loss, grads = loss_and_grad_fn(model, X, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
    return loss.item()

def main(args):
    seed = args.seed
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay

    np.random.seed(seed)

    print("Loading Dataset")
    train_images, train_labels, test_images, test_labels = map(
        mx.array, getattr(mnist, args.dataset)()
    )
    print("Dataset Loaded")

    model = create_model(args)

    mx.eval(model.parameters())
    print_trainable_parameters(model)

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)

    print(f"starting to train for {num_epochs} epochs")
    for e in range(args.num_epochs):
        current_epoch = e + 1
        tic = time.perf_counter()
        if args.train_batched:
            loss = train_epoch_batched(model, optimizer, train_images, train_labels, args.batch_size, loss_and_grad_fn=loss_and_grad_fn)
        else:
            loss = train_epoch(model, optimizer, train_images, train_labels, args.batch_size, loss_and_grad_fn=loss_and_grad_fn)
        toc = time.perf_counter()
        logging.info(f"Epoch {current_epoch}: Train Loss: {loss:.4f}, Time {toc - tic:.3f} (s)")
        print(f"Epoch {current_epoch}: Train Loss: {loss:.4f}, Time {toc - tic:.3f} (s)")

        if current_epoch == 1 or current_epoch % args.eval_report_count == 0 or e == args.num_epochs:
            tic = time.perf_counter()
            accuracy = batched_eval_fn(model, test_images, test_labels, args.batch_size)
            toc = time.perf_counter()
            logging.info(f"Epoch {current_epoch}: Test accuracy {accuracy.item():.8f}, Time {toc - tic:.3f} (s)")
            print(f"Epoch {current_epoch}: Test accuracy {accuracy.item():.8f}, Time {toc - tic:.3f} (s)")

    mx.eval(model.parameters(), optimizer.state)

    save_dir = create_save_directory(args.save_path)

    save_model(model=model, save_path=save_dir)
    save_config(args, save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train KAN on MNIST with MLX.")
    parser.add_argument("--cpu", action="store_true", help="Use the CPU instead og Metal GPU backend.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "fashion_mnist"],
        help="The dataset to use.",
    )
    parser.add_argument("--use-kan-convolution", action="store_true", help="Use the Convolution KAN architecture.")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of layers in the model.")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Number of hidden units in each layer.")
    parser.add_argument("--num-classes", type=int, default=10, help="Number of output classes.")
    parser.add_argument("--in-features", type=int, default=28, help="Number input features.")
    parser.add_argument("--out-features", type=int, default=28, help="Number output features.")
    parser.add_argument("--num-epochs", type=int, default=1, help="Number of epochs to train.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate for the optimizer.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for the optimizer.")
    parser.add_argument("--eval-report-count", type=int, default=10, help="Number of epochs to report validations / test accuracy.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument("--train-batched", action="store_true", help="Train the model with batching.")
    parser.add_argument("--save-path", type=str, default="trained_kan_model", help="Directory to save the model and config.")
    args = parser.parse_args()

    if args.cpu:
        mx.set_default_device(mx.cpu)

    main(args)