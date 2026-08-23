#!/usr/bin/env python3
"""Map a physical NVIDIA index to VirtualGL's EGL device identifier."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
from pathlib import Path


EGL_EXTENSIONS = 0x3055
EGL_PLATFORM_DEVICE_EXT = 0x313F
EGL_DRM_DEVICE_FILE_EXT = 0x3233
EGL_CUDA_DEVICE_NV = 0x323A


def run(command: list[str]) -> str:
    process = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if process.returncode:
        raise SystemExit(
            f"command failed ({process.returncode}): {' '.join(command)}\n{process.stderr}"
        )
    return process.stdout


def load_extension(egl: ctypes.CDLL, name: bytes, function_type):
    address = egl.eglGetProcAddress(name)
    if not address:
        raise SystemExit(f"EGL extension function is unavailable: {name.decode()}")
    return function_type(address)


def enumerate_egl_devices() -> list[dict[str, object]]:
    nvidia_vendor = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
    if not Path(nvidia_vendor).is_file():
        raise SystemExit(f"NVIDIA EGL vendor manifest is missing: {nvidia_vendor}")
    os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = nvidia_vendor
    egl = ctypes.CDLL("libEGL.so.1")
    egl.eglGetProcAddress.argtypes = [ctypes.c_char_p]
    egl.eglGetProcAddress.restype = ctypes.c_void_p
    egl.eglInitialize.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]
    egl.eglInitialize.restype = ctypes.c_uint
    egl.eglTerminate.argtypes = [ctypes.c_void_p]
    egl.eglTerminate.restype = ctypes.c_uint
    egl.eglQueryString.argtypes = [ctypes.c_void_p, ctypes.c_int]
    egl.eglQueryString.restype = ctypes.c_char_p

    query_devices = load_extension(
        egl,
        b"eglQueryDevicesEXT",
        ctypes.CFUNCTYPE(
            ctypes.c_uint,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_int),
        ),
    )
    get_platform_display = load_extension(
        egl,
        b"eglGetPlatformDisplayEXT",
        ctypes.CFUNCTYPE(
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_ssize_t),
        ),
    )
    query_device_attrib = load_extension(
        egl,
        b"eglQueryDeviceAttribEXT",
        ctypes.CFUNCTYPE(
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_ssize_t),
        ),
    )
    query_device_string = load_extension(
        egl,
        b"eglQueryDeviceStringEXT",
        ctypes.CFUNCTYPE(ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int),
    )

    count = ctypes.c_int()
    if not query_devices(0, None, ctypes.byref(count)) or count.value < 1:
        raise SystemExit("no EGL devices found")
    raw_devices = (ctypes.c_void_p * count.value)()
    if not query_devices(count.value, raw_devices, ctypes.byref(count)):
        raise SystemExit("could not enumerate EGL devices")

    valid_devices = []
    for raw_device in raw_devices[: count.value]:
        display = get_platform_display(EGL_PLATFORM_DEVICE_EXT, raw_device, None)
        major = ctypes.c_int()
        minor = ctypes.c_int()
        if not display or not egl.eglInitialize(display, ctypes.byref(major), ctypes.byref(minor)):
            continue
        extensions_bytes = egl.eglQueryString(display, EGL_EXTENSIONS)
        extensions = extensions_bytes.decode() if extensions_bytes else ""
        egl.eglTerminate(display)
        cuda_index = ctypes.c_ssize_t(-1)
        has_cuda_index = bool(
            query_device_attrib(raw_device, EGL_CUDA_DEVICE_NV, ctypes.byref(cuda_index))
        )
        dri_bytes = query_device_string(raw_device, EGL_DRM_DEVICE_FILE_EXT)
        valid_devices.append(
            {
                "egl_device": f"egl{len(valid_devices)}",
                "cuda_device": cuda_index.value if has_cuda_index else None,
                "dri_device": dri_bytes.decode() if dri_bytes else None,
                "egl_version": f"{major.value}.{minor.value}",
                "extensions": extensions.split(),
            }
        )
    return valid_devices


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True, help="Physical nvidia-smi GPU index")
    parser.add_argument(
        "--eglinfo",
        type=Path,
        default=Path(
            "/data/yukun/.cache/racer/virtualgl-3.1.5-root/opt/VirtualGL/bin/eglinfo"
        ),
        help="Pinned VirtualGL eglinfo; its presence binds the mapping to the installed release.",
    )
    parser.add_argument("--print-device", action="store_true")
    args = parser.parse_args()

    if args.gpu < 0:
        raise SystemExit("--gpu must be nonnegative")
    if not args.eglinfo.is_file():
        raise SystemExit(f"eglinfo is missing: {args.eglinfo}")

    gpu_indices = {
        int(line.split(",", 1)[0].strip())
        for line in run(
            ["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader,nounits"]
        ).splitlines()
    }
    if args.gpu not in gpu_indices:
        raise SystemExit(f"nvidia-smi GPU index not found: {args.gpu}")

    mappings = enumerate_egl_devices()
    target = [row for row in mappings if row["cuda_device"] == args.gpu]
    if len(target) != 1:
        raise SystemExit(
            f"expected one EGL mapping for NVIDIA GPU {args.gpu}, found {len(target)}"
        )
    required = {"EGL_KHR_no_config_context", "EGL_KHR_surfaceless_context"}
    missing = required.difference(target[0]["extensions"])
    if missing:
        raise SystemExit(f"selected EGL device lacks required extensions: {sorted(missing)}")

    report = {"requested_gpu": args.gpu, "selected": target[0], "all_devices": mappings}
    if args.print_device:
        print(target[0]["egl_device"])
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
