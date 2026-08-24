"""Which physical GPU does each MUJOCO_EGL_DEVICE_ID land on?

EGL enumerates devices in its own order, unrelated to CUDA's. Setting
CUDA_VISIBLE_DEVICES does not move the renderer, and MUJOCO_EGL_DEVICE_ID=0 is
not necessarily GPU 0 -- so a MuJoCo eval can quietly render on a card you meant
to keep free.

This asks EGL directly, via the EGL_CUDA_DEVICE_NV attribute that NVIDIA's EGL
exposes on each device handle.

    python scripts/egl_device_map.py

Read-only: it creates and destroys nothing beyond device queries.
"""

from __future__ import annotations

import ctypes
import ctypes.util

EGL_PLATFORM_DEVICE_EXT = 0x313F  # noqa: F841  (kept for reference)
EGL_CUDA_DEVICE_NV = 0x323A


def egl_to_cuda() -> dict[int, int]:
    """{MUJOCO_EGL_DEVICE_ID: cuda device index} for devices that expose one."""
    path = ctypes.util.find_library("EGL") or "libEGL.so.1"
    egl = ctypes.CDLL(path)
    get_proc = egl.eglGetProcAddress
    get_proc.restype = ctypes.c_void_p
    get_proc.argtypes = [ctypes.c_char_p]

    addr_devices = get_proc(b"eglQueryDevicesEXT")
    addr_attrib = get_proc(b"eglQueryDeviceAttribEXT")
    if not addr_devices or not addr_attrib:
        return {}

    EGLDeviceEXT = ctypes.c_void_p
    query_devices = ctypes.CFUNCTYPE(
        ctypes.c_uint, ctypes.c_int, ctypes.POINTER(EGLDeviceEXT), ctypes.POINTER(ctypes.c_int)
    )(addr_devices)
    query_attrib = ctypes.CFUNCTYPE(
        ctypes.c_uint, EGLDeviceEXT, ctypes.c_int, ctypes.POINTER(ctypes.c_ssize_t)
    )(addr_attrib)

    count = ctypes.c_int(0)
    if not query_devices(0, None, ctypes.byref(count)):
        return {}
    devices = (EGLDeviceEXT * count.value)()
    if not query_devices(count.value, devices, ctypes.byref(count)):
        return {}

    mapping: dict[int, int] = {}
    for i in range(count.value):
        value = ctypes.c_ssize_t(0)
        if query_attrib(devices[i], EGL_CUDA_DEVICE_NV, ctypes.byref(value)):
            mapping[i] = int(value.value)
    return mapping


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--for_cuda",
        type=int,
        default=None,
        help="print just the MUJOCO_EGL_DEVICE_ID that renders on this nvidia-smi index",
    )
    args = ap.parse_args()

    if args.for_cuda is not None:
        mapping = egl_to_cuda()
        for egl_id, cuda_id in sorted(mapping.items()):
            if cuda_id == args.for_cuda:
                print(egl_id)
                return
        raise SystemExit(f"no EGL device maps to CUDA {args.for_cuda}; got {mapping}")

    path = ctypes.util.find_library("EGL") or "libEGL.so.1"
    egl = ctypes.CDLL(path)

    get_proc = egl.eglGetProcAddress
    get_proc.restype = ctypes.c_void_p
    get_proc.argtypes = [ctypes.c_char_p]

    query_devices_addr = get_proc(b"eglQueryDevicesEXT")
    query_attrib_addr = get_proc(b"eglQueryDeviceAttribEXT")
    if not query_devices_addr:
        raise SystemExit("eglQueryDevicesEXT unavailable -- cannot enumerate EGL devices")

    EGLDeviceEXT = ctypes.c_void_p
    query_devices = ctypes.CFUNCTYPE(
        ctypes.c_uint, ctypes.c_int, ctypes.POINTER(EGLDeviceEXT), ctypes.POINTER(ctypes.c_int)
    )(query_devices_addr)

    count = ctypes.c_int(0)
    if not query_devices(0, None, ctypes.byref(count)):
        raise SystemExit("eglQueryDevicesEXT failed to report a device count")
    n = count.value
    print(f"EGL reports {n} device(s)\n")

    devices = (EGLDeviceEXT * n)()
    if not query_devices(n, devices, ctypes.byref(count)):
        raise SystemExit("eglQueryDevicesEXT failed to fill the device list")

    query_attrib = None
    if query_attrib_addr:
        query_attrib = ctypes.CFUNCTYPE(
            ctypes.c_uint, EGLDeviceEXT, ctypes.c_int, ctypes.POINTER(ctypes.c_ssize_t)
        )(query_attrib_addr)

    print(f"{'MUJOCO_EGL_DEVICE_ID':>21}  {'-> CUDA device':>15}")
    for i in range(count.value):
        cuda_index = "unknown"
        if query_attrib is not None:
            value = ctypes.c_ssize_t(0)
            if query_attrib(devices[i], EGL_CUDA_DEVICE_NV, ctypes.byref(value)):
                cuda_index = str(value.value)
        print(f"{i:>21}  {cuda_index:>15}")

    print(
        "\nCUDA device index here matches `nvidia-smi -i <n>` when CUDA_VISIBLE_DEVICES is unset.\n"
        "Pick the MUJOCO_EGL_DEVICE_ID whose CUDA device is the card you want to render on."
    )


if __name__ == "__main__":
    main()
