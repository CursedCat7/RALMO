"""Model abstractions for RALMO — draft, target, and base interfaces.

Note: DraftModel and TargetModel require llama-cpp-python to be installed.
Import them directly from their modules when needed, or use the lazy
re-exports below which will raise ImportError if llama_cpp is not available.
"""

from ralmo_core.models.base_model import BaseModel, GenerationResult, KVSnapshot

__all__ = ["BaseModel", "GenerationResult", "KVSnapshot", "DraftModel", "TargetModel"]


def __getattr__(name: str):  # type: ignore[no-untyped-def]
    """Lazy imports for models that depend on llama_cpp."""
    if name == "DraftModel":
        from ralmo_core.models.draft_model import DraftModel

        return DraftModel
    if name == "TargetModel":
        from ralmo_core.models.target_model import TargetModel

        return TargetModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
