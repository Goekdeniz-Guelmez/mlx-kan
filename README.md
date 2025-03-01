# KAN: Kolmogorov–Arnold Networks in MLX for Apple silicon

Welcome to my implementation of Kolmogorov–Arnold Networks (KAN), meticulously optimized for Apple Silicon using the MLX framework. This Python package leverages the exceptional computational capabilities of Apple’s M1 chip and later versions, providing an advanced, efficient, and scalable solution for developing, training, and evaluating KAN models. The package is designed to facilitate seamless integration with popular datasets such as MNIST and Fashion MNIST, showcasing the versatility and robustness of KANs in various machine learning tasks.

Kolmogorov–Arnold Networks represent a sophisticated approach to neural network design, incorporating innovative mathematical principles to enhance learning efficiency and model performance. This implementation is grounded in cutting-edge research, as detailed in the Kolmogorov–Arnold Networks paper, and is tailored to exploit the unique architecture of Apple’s silicon, ensuring optimal performance and resource utilization.


Based on the [paper](https://arxiv.org/pdf/2404.19756)

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Arguments](#arguments)
  - [Examples](#examples)
- [Model Architecture](#model-architecture)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Usage with PyPi

install the Package:

```sh
pip install mlx-kan
```

Example usage in Python:

```python
from mlx_kan.kan import KAN

# Initialize and use KAN
kan_model = KAN([in_features * out_features] + [hidden_dim] * (num_layers - 1) + [num_classes])
```

### ModelArgs

The `ModelArgs` class for the basic KAN model defines the arguments for configuring the KAN model:

```python
class ModelArgs:
    layers_hidden: Optional[List[int]] = None
    model_type: str = "KAN"
    num_layers: int = 2
    in_features: int = 28
    out_features: int = 28
    hidden_dim: int = 64
    num_classes: int = 10
    grid_size: int = 5
    spline_order: float = 3
    scale_noise: float = 0.1
    scale_base: float = 1.0
    scale_spline: float = 1.0
    hidden_act = nn.SiLU
    grid_eps: float = 0.02
    grid_range = [-1, 1]
```

### TrainArgs

The `TrainArgs` class defines the training configuration:

```python
@dataclass
class TrainArgs:
    train_algorithm: str = "simple"
    dataset: str = "custom"
    max_steps: int = 0
    epochs: int = 2
    max_train_batch_size: int = 32
    max_val_batch_size: int = 32
    max_test_batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    clip_grad_norm: bool = False
    save_path: str = "./models"
```

### SimpleTrainer

The `SimpleTrainer` function facilitates model training with the specified arguments and datasets:

```python
def SimpleTrainer(
    model: nn.Module,
    args: TrainArgs,
    train_set: Optional[tuple] = None,
    validation_set: Optional[tuple] = None,
    test_set: Optional[tuple] = None,
    validation_interval: Optional[int] = None,
    logging_interval: int = 10
)
```

### **You can find additional example files in the mlx-kan/examples directory to help you get started with various configurations and training setups.**

### If you just want to try out a quick and simple training session:

```sh
python -m mlx-kan.quick_scripts.quick_train --help
```

```sh
python -m mlx-kan.quick_scripts.quick_train --num-layers 2 --hidden-dim 64 --num-epochs 2 --batch-size 14 --seed 42 --clip-grad-norm
```

## New MLP Architectures

The following classes define different sizes of MLP architectures using `KANLinear` layers. You can import them via:

```python
from mlx_kan.kan.architectures.KANMLP import LlamaKANMLP, SmallKANMLP, MiddleKANMLP, BigKANMLP
```

### Parameters

- `in_features`: The number of input features.
- `hidden_dim`: The number of hidden units in each layer.
- `out_features`: The number of output features.
- `grid_size`: The size of the grid used in the `KANLinear` layer. Default is `5`.
- `spline_order`: The order of the spline used in the `KANLinear` layer. Default is `3`.
- `scale_noise`: The noise scaling factor. Default is `0.1`.
- `scale_base`: The base scaling factor. Default is `1.0`.
- `scale_spline`: The spline scaling factor. Default is `1.0`.
- `enable_standalone_scale_spline`: Whether to enable standalone scaling for the spline. Default is `True`.
- `hidden_act`: The activation function used in hidden layers. Default is `nn.SiLU`.
- `grid_eps`: The epsilon value for the grid. Default is `0.02`.
- `grid_range`: The range of the grid. Default is `[-1, 1]`.

### `SmallKANMLP` Class

The `SmallKANMLP` class consists of two `KANLinear` layers. It is designed for small-scale models.

### `MiddleKANMLP` Class

The `MiddleKANMLP` class consists of three `KANLinear` layers. It is designed for medium-scale models.

### `BigKANMLP` Class

The `BigKANMLP` class consists of four `KANLinear` layers. It is designed for large-scale models.

### `LlamaKANMLP` Class

The `LlamaKANMLP` class consists of three `KANLinear` layers configured in a the same manner Llama's MLP layer is configured. It is designed for models requiring a unique layer arrangement.


## Training

if you wish to train the MLP's, you need to update the grid points at the end of each epoch.

```python
# Update grid points here
for name, layer in model.__dict__.items():
    if isinstance(layer, KANLinear):
        with mx.no_grad():
            layer.update_grid(train_set)
```

### Training step example

```python
def train(model, train_set, train_labels, num_epochs=100):
    optimizer = optim.AdamW(learning_rate=0.0004, weight_decay=0.003)  # Initialize a new optimizer for each model
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    # For 1 step
    loss, grads = loss_and_grad_fn(model, train_set, train_labels)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)
    avg_loss = total_loss += loss.item()

    # Update grid points here
    for name, layer in model.__dict__.items():
        if isinstance(layer, KANLinear):
            with mx.no_grad():
                layer.update_grid(train_set)
````

-----
<br>
<br>
<br>

# From source

## Clone this Repo

To run this example, you need to have Python and the necessary libraries installed. Follow these steps to set up your environment:

1. Clone the repository:

```bash
git clone https://github.com/Goekdeniz-Guelmez/mlx-kan.git
cd mlx-kan
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

You can run the script `main.py` to train the KAN model on the MNIST dataset. The script supports various command-line arguments for configuration.

### Arguments

- `--cpu`: Use the Metal back-end.
- `--clip-grad-norm`: Use gradient clipping to prevent the gradients from becoming too large. Default is `False`.
- `--dataset`: The dataset to use (`mnist` or `fashion_mnist`). Default is `mnist`.
- `--num_layers`: Number of layers in the model. Default is `2`.
- `--in-features`: Number input features. Default is `28`.
- `--out-features`: Number output features. Default is `28`.
- `--num-classes`: Number of output classes. Default is `10`.
- `--hidden_dim`: Number of hidden units in each layer. Default is `64`.
- `--num_epochs`: Number of epochs to train. Default is `10`.
- `--batch_size`: Batch size for training. Default is `64`.
- `--learning_rate`: Learning rate for the optimizer. Default is `1e-3`.
- `--weight-decay`: Weight decay for the optimizer. Default is `1e-4`.
- `--eval-report-count`: Number of epochs to report validations / test accuracy values. Default is `10`.
- `--save-path`: Path with the model name where the trained KAN model will be saved. Default is `traned_kan_model.safetensors`.
- `--train-batched`: Use batch training instead of single epoch. Default is `False`.
- `--seed`: Random seed for reproducibility. Default is `0`.

### Examples

#### Find all Arguments wioth descriptions

```sh
python -m quick_scripts.quick_train --help
```

#### Basic Usage

Train the KAN model on the MNIST dataset with default settings:

```sh
python -m quick_scripts.quick_train --dataset mnist
```

#### Custom Configuration

Train the KAN model with a custom configuration:

```sh
python quick_train.py --dataset fashion_mnist --hidden-dim 256 --num-epochs 30
 --batch-size 256 --learning-rate 5e-4 --weight-decay 1e-6
 --clip-grad-norm --eval-interval 3 --seed 42
 --save-path "models/fashion_mnist_large"
```

#### Using CPU

Train the KAN model using the CPU backend:

```sh
python -m quick_scripts.quick_train --cpu --dataset mnist
```

## Model Architecture

The `KAN` (Kolmogorov–Arnold Networks) class defines the model architecture. The network consists of multiple `KANLinear` layers, each defined by the provided parameters. The number of layers and the hidden dimension size can be configured via command-line arguments.

### Example Model Initialization

```python
layers_hidden = [in_features * out_features] + [hidden_dim] * (num_layers - 1) + [num_classes]
model = KAN(layers_hidden)
```

----

<br>

### KAN Class

The `KAN` class initializes a sequence of `KANLinear` layers based on the provided hidden layers configuration. Each layer performs linear transformations with kernel attention mechanisms.

```python
class KAN(nn.Module):
    def __init__(self, layers_hidden, grid_size=5, spline_order=3, scale_noise=0.1, scale_base=1.0, scale_spline=1.0, base_activation=nn.SiLU, grid_eps=0.02, grid_range=[-1, 1]):
        super().__init__()
        self.layers = []
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features, out_features, grid_size, spline_order, scale_noise, scale_base, scale_spline, base_activation, grid_eps, grid_range
                )
            )
    def __call__(self, x, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x
    
    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return mx.add(*(
            layer.regularization_loss(regularize_activation, regularize_entropy) 
            for layer in self.layers
        ))
```

## Contributing

Contributions are welcome! If you have any suggestions or improvements, feel free to open an issue or submit a pull request.

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add new feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Create a new Pull Request.

## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.

Made with love by Gökdeniz Gülmez.

<br>
<br>
<br>
<br>
<br>

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Goekdeniz-Guelmez/mlx-kan.git&type=Date)](https://star-history.com/#Goekdeniz-Guelmez/mlx-kan.git&Date)

<br>
<br>

---

## Citing mlx-kan

The mlx-kan software suite was developed by Gökdeniz Gülmez. If you find
mlx-kan useful in your research and wish to cite it, please use the following
BibTex entry:

```text
@software{
  mlx-kan,
  author = {Gökdeniz Gülmez},
  title = {{mlx-kan}: KAN: Kolmogorov–Arnold Networks in MLX for Apple silicon},
  url = {https://github.com/Goekdeniz-Guelmez/mlx-kan},
  version = {0.2.5},
  year = {2024},
}
```
