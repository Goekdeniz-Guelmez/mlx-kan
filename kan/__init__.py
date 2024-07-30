from .version import __version__

def __getattr__(name):
    if name in ("KANLinear", "KAN"):
        from .kan import KANLinear, KAN
        return locals()[name]
    if name in ("LlamaKANMLP", "SmallKANMLP", "MiddleKANMLP", "BigKANMLP"):
        from .architectures.KANMLP import LlamaKANMLP, SmallKANMLP, MiddleKANMLP, BigKANMLP
        return locals()[name]

__all__ = ["KANLinear", "KAN", "LlamaKANMLP", "SmallKANMLP", "MiddleKANMLP", "BigKANMLP"]