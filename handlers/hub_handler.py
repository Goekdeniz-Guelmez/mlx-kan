from typing import Tuple, Path
from textwrap import dedent

import mlx.nn as nn

from model_handler import load_config, load_model


def pull_from_hub(model_path: Path, lazy: bool = False) -> Tuple[nn.Module, dict]:
    model = load_model(model_path, lazy)
    config = load_config(model_path)
    return model, config


def push_to_hub(path: str, upload_repo: str):
    """
    Uploads the model to Hugging Face hub.

    Args:
        path (str): Local path to the model.
        upload_repo (str): Name of the HF repo to upload to.
    """
    import os

    from huggingface_hub import HfApi, ModelCard, logging

    from . import __version__

    card = ModelCard.load(upload_repo)
    card.data.tags = ["mlx-kan"] if card.data.tags is None else card.data.tags + ["mlx-kan"]
    card.data.base_model = upload_repo
    card.text = dedent(
        f"""
        # {upload_repo}

        The Model [{upload_repo}](https://huggingface.co/{upload_repo}) was created with [mlx-kan](https://github.com/Goekdeniz-Guelmez/mlx-kan) in MLX format using version **{__version__}**.

        ```bash
        pip install mlx-kan
        ```
        """
    )
    card.save(os.path.join(path, "README.md"))

    logging.set_verbosity_info()

    api = HfApi()
    api.create_repo(repo_id=upload_repo, exist_ok=True)
    api.upload_folder(
        folder_path=path,
        repo_id=upload_repo,
        repo_type="model",
        multi_commits=True,
        multi_commits_verbose=True,
    )
    print(f"Upload successful, go to https://huggingface.co/{upload_repo} for details.")