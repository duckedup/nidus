"""The SDK version — the single place it is written down.

DO NOT HAND-EDIT. `.github/workflows/sdk-py-release.yml` stamps this file from
`Cargo.toml`'s `version` at release time and commits the stamp back to `main`, the
same way the Helm chart's versions are derived. The crate and every SDK ship at one
version; hand-editing here would fork that and leave a clone's `nidus.__version__`
disagreeing with what is on PyPI.

`pyproject.toml` reads it via `[tool.hatch.version] path`, so there is exactly one
string to stamp.
"""

from __future__ import annotations

__version__ = "0.40.0"
