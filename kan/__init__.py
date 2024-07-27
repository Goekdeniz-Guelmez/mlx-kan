# Copyright © 2024 Gökdeniz Gülmez

from .version import __version__

def __getattr__(name):
    if name in ("KANLinear", "KAN"):
        from .kan import KANLinear, KAN
        return locals()[name]
    elif name in ("KAN_Convolutional_Layer", "KAN_Convolution"):
        from .kan_convolution import KAN_Convolutional_Layer, KAN_Convolution
        return locals()[name]
    elif name in ("KKAN_Convolutional_Network"):
        from .architectures import KKAN_Convolutional_Network, CKAN, KANC_MLP
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "KANLinear",
    "KAN",
    "KAN_Convolutional_Layer",
    "KAN_Convolution",
    "KKAN_Convolutional_Network",
    "CKAN",
    "KANC_MLP"
]