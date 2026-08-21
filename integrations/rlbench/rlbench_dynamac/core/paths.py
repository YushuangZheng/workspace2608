"""Stable repository paths for modules nested below ``rlbench_dynamac``."""

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
INTEGRATION_ROOT = PACKAGE_ROOT.parent
REPOSITORY_ROOT = INTEGRATION_ROOT.parents[1]
