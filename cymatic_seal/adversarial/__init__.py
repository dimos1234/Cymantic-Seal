def __getattr__(name):
    """Lazy import so 'import torch' only happens when the engine is used."""
    if name == "AdversarialEngine":
        from .engine import AdversarialEngine
        return AdversarialEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
