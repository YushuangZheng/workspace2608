# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Python module serving as a project/extension template.
"""

# Isaac Sim modules such as ``pxr`` only become importable after AppLauncher
# starts the application.  Keep data and policy modules usable by ordinary
# Python processes while preserving automatic task registration in simulation.
try:
    from .tasks import *
    from .ui_extension_example import *
except ModuleNotFoundError as error:
    if error.name.split(".")[0] not in {"carb", "isaacsim", "omni", "pxr"}:
        raise
