from typing import Optional
from tqdm import tqdm
import time

import numpy as np

import mlx.optimizers as optimizer
import mlx.core as mx
import mlx.nn as nn

from trainer.trainer_args import TrainArgs
import quick_scripts.mnist as mnist
from quick_scripts.utils import get_parameters, save_model, save_config, create_save_directory


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


def train_epoch_batched(
        model: nn.Module,
        optimizer: optimizer.Optimizer,
        train_inputs: mx.array,
        train_labels: mx.array,
        batch_size: int,
        loss_and_grad_fn,
        clip_grad_norm: bool = False
    ) -> float:
    total_loss = 0
    num_batches = 0

    model.train()
    for X, y in tqdm(batch_iterate(batch_size, train_inputs, train_labels), total=len(train_inputs) // batch_size, desc='Training'):
        loss, grads = loss_and_grad_fn(model, X, y)
        if mx.isnan(loss).any():
            raise ValueError("Encountered NaN in loss")
        if clip_grad_norm:
            optimizer.clip_grad_norm(grads, max_norm=1.0)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        total_loss += loss.item()
        num_batches += 1
    return total_loss / num_batches


def train_step_batched(
        model: nn.Module,
        optimizer: optimizer.Optimizer,
        train_inputs: mx.array,
        train_labels: mx.array,
        batch_size: int,
        loss_and_grad_fn,
        clip_grad_norm: bool = False
    ) -> float:
    total_loss = 0
    num_batches = 0

    model.train()
    batch_iter = batch_iterate(batch_size, train_inputs, train_labels)
    try:
        X, y = next(batch_iter)
        loss, grads = loss_and_grad_fn(model, X, y)
        if mx.isnan(loss).any():
            raise ValueError("Encountered NaN in loss")
        if clip_grad_norm:
            optimizer.clip_grad_norm(grads, max_norm=1.0)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        total_loss += loss.item()
        num_batches += 1
    except StopIteration:
        pass

    return total_loss / num_batches


def SimpleTrainer(
        model: nn.Module,
        dataset_type: str = "mnist", # mnist, fashion_mnist, or custom
        train_set: Optional[tuple] = None,
        validation_set: Optional[tuple] = None,
        test_set: Optional[tuple] = None,
        max_steps: int = 0,
        epochs: int = 2,
        max_train_batch_size: int = 64,
        max_val_batch_size: int = 64,
        max_test_batch_size: int = 64,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        clip_grad_norm: bool = False,
        save_path: str = './models'
    ):

    if dataset_type != "custom":
        train_images, train_labels, test_images, test_labels = map(mx.array, getattr(mnist, dataset_type)())
        validation_images, validation_labels = test_images, test_labels
    else:
        train_images, train_labels = map(mx.array, train_set)
        validation_images, validation_labels = map(mx.array, validation_set)
        test_images, test_labels = map(mx.array, test_set)

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    opt = optimizer.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)

    print(f"\nStarting to train model for {epochs if max_steps == 0 else max_steps} {'epochs' if max_steps == 0 else 'steps'}")
    
    if max_steps > 0:
        # Training using steps
        current_step = 0
        with tqdm(total=max_steps, desc="Training Steps") as pbar:
            while current_step < max_steps:
                tic = time.perf_counter()
                loss = train_step_batched(model, opt, train_images, train_labels, max_train_batch_size, loss_and_grad_fn=loss_and_grad_fn, clip_grad_norm=clip_grad_norm)
                toc = time.perf_counter()
                current_step += 1
                pbar.set_postfix({"Train Loss": f"{loss:.4f}", "Time (s)": f"{toc - tic:.3f}"})
                pbar.update(1)

                if current_step % 5 == 0 or current_step == max_steps:
                    tic = time.perf_counter()
                    val_accuracy = batched_eval_fn(model, validation_images, validation_labels, max_val_batch_size)
                    toc = time.perf_counter()
                    tqdm.write(f"-----Step {current_step}: Validation accuracy {val_accuracy.item():.8f}, Time {toc - tic:.3f} (s)")
    else:
        # Training using epochs
        for e in range(epochs):
            current_epoch = e + 1
            tic = time.perf_counter()
            loss = train_epoch_batched(model, opt, train_images, train_labels, max_train_batch_size, loss_and_grad_fn=loss_and_grad_fn, clip_grad_norm=clip_grad_norm)
            toc = time.perf_counter()
            print(f"Epoch {current_epoch}: Train Loss: {loss:.4f}, Time {toc - tic:.3f} (s)")

            if current_epoch == 1 or current_epoch % 5 == 0 or current_epoch == epochs:
                tic = time.perf_counter()
                val_accuracy = batched_eval_fn(model, validation_images, validation_labels, max_val_batch_size)
                toc = time.perf_counter()
                print(f"-----Epoch {current_epoch}: Validation accuracy {val_accuracy.item():.8f}, Time {toc - tic:.3f} (s)")

    # Evaluate on the test set
    tic = time.perf_counter()
    test_accuracy = batched_eval_fn(model, test_images, test_labels, max_test_batch_size)
    toc = time.perf_counter()
    print(f"Test accuracy {test_accuracy.item():.8f}, Time {toc - tic:.3f} (s)")

    mx.eval(model.parameters(), opt.state)
    save_dir = create_save_directory(save_path)
    save_model(model=model, save_path=save_dir)
    save_config({
        "dataset_type": dataset_type,
        "epochs": epochs,
        "steps": max_steps,
        "batch_size": max_train_batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "clip_grad_norm": clip_grad_norm
    }, save_dir)