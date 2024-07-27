# Copyright © 2024 Gökdeniz Gülmez

# TODO
from .version import __version__

def __getattr__(name):
    if name in ("KANLinear", "KAN"):
        from .kan import KANLinear, KAN
        return locals()[name]
    elif name in ("KAN_Convolutional_Layer", "KAN_Convolution"):
        from .convolution.kanConvolution import KAN_Convolutional_Layer, KAN_Convolution
        return locals()[name]
    elif name in ("KKAN_Convolutional_Network"):
        from .convolution.kkan import KKAN_Convolutional_Network
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["KANLinear", "KAN", "KAN_Convolutional_Layer", "KAN_Convolution", "KKAN_Convolutional_Network"]