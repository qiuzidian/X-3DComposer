"""X-3DComposer: feed-forward 3D scene composition with any assets from any views.

    from x3dcomposer import Asset, SceneObservation, X3DComposer, compose

The public names are imported lazily, so the NumPy-only data tools work without PyTorch installed.
"""

from importlib import import_module

__version__ = "1.0.0"

_EXPORTS = {
    "X3DComposer": "models.x3dcomposer", "ModelConfig": "config", "TrainConfig": "config", "load_config": "config",
    "Asset": "data.assets", "SceneObservation": "inference.pipeline", "Layout": "inference.pipeline",
    "compose": "inference.pipeline", "prepare": "inference.pipeline", "read_out": "inference.readout",
    "save_layout": "inference.export", "export_glb": "inference.export",
}
__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module 'x3dcomposer' has no attribute {name!r}")
    return getattr(import_module(f"{__name__}.{_EXPORTS[name]}"), name)
