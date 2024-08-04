# TODO's

- Adding a better training algorythm.
- Adding custom Tokenizer setting.
- Adding custom Dataset with preperty.
- Adding a from_pretrained method in the KAN model class
- Adding the ModelArgs to kan/args.py
- Maybe Adding LLM's as a KAN instead of MLP kan/llms/*.py -> llama.py, ... mistral.py, ...

```python
from mlx_kan.kan import KAN
from mlx_kan.kan.args import ModelArgs
from mlx_kan.trainer.simpletrainer import SimpleTrainer

kan_model = KAN(ModelArgs)

SimpleTrainer(
    model=kan_model,
    train=train,
    validation=validation,
    test=test,
    max_steps=1000,
    epochs=2,
    max_train_batch_size=ModelArgs.max_batch_size,
    max_val_batch_size=ModelArgs.max_batch_size,
    max_test_batch_size=ModelArgs.max_batch_size
)
```
