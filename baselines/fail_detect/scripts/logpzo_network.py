#!/usr/bin/env python3
"""Load the released logpZO network from its actual upstream CFM module."""

import importlib
import sys
from pathlib import Path


def load_logpzo_module(upstream):
    upstream = Path(upstream).resolve()
    uq_root = upstream / "UQ_baselines"
    expected_module = (uq_root / "CFM/net_CFM.py").resolve()
    if not expected_module.is_file():
        raise RuntimeError("missing released logpZO network module: {}".format(expected_module))
    for path in (upstream, uq_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    module = importlib.import_module("CFM.net_CFM")
    actual_module = Path(module.__file__).resolve()
    if actual_module != expected_module:
        raise RuntimeError(
            "resolved CFM.net_CFM from the wrong checkout: {} != {}".format(
                actual_module, expected_module
            )
        )
    return module


def build_logpzo_network(upstream, input_dim=20):
    return load_logpzo_module(upstream).get_unet(input_dim)
