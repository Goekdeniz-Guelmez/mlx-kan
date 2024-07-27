from .version import __version__

def __getattr__(name):
    if name in ("KANLinear", "KAN"):
        from .kan import KANLinear, KAN
        return locals()[name]
    elif name in ("KAN_Convolutional_Layer", "KAN_Convolution"):
        from .kan_convolution.kanConvolution import KAN_Convolutional_Layer, KAN_Convolution
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["KANLinear", "KAN", "KAN_Convolutional_Layer", "KAN_Convolution"]