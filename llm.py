import mlx.core as mx

from kan.architectures.qwen2 import ModelArgs, Model

model = Model(ModelArgs)
print(model)

input = mx.array([[1, 2, 3], [4, 5, 6]])
output = model(input)
print(f"Output shape: {output.shape}")
print(f"Output: {output}")