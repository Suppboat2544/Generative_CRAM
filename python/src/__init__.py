"""
CRAM Generator source package.

Keep package import lightweight so command entrypoints can run even when
optional training dependencies (for example DGL) are not installed.
Import submodules directly where needed, e.g. ``from src.generators import ...``.
"""

__all__ = ["models", "generators", "features"]
