"""NVML library reads (SPEC 6d gpu object, 2e startup free-memory gate).

One library-call reader on device index 0 only — never a shell-out to
nvidia-smi. A failed read returns None; no value is ever guessed and no
default byte count exists anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Device0Read:
    available: bool
    device_count: int
    name: str
    memory_total_bytes: int | None
    memory_used_bytes: int | None


def read_device0() -> Device0Read | None:
    """Open NVML and read device 0 (SPEC 6d). None when NVML is missing,
    permission is denied, or zero devices — the caller decides the consequence;
    this function invents nothing."""
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            count = int(pynvml.nvmlDeviceGetCount())
            if count < 1:
                return None
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            raw_name = pynvml.nvmlDeviceGetName(handle)
            name = raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return Device0Read(
                available=True,
                device_count=count,
                name=name,
                memory_total_bytes=int(mem.total),
                memory_used_bytes=int(mem.used),
            )
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
    except Exception:
        return None


def gpu_inventory() -> dict:
    """The closed `gpu` object of /v1/runtime (SPEC 6d): available false and
    null memory fields when NVML cannot be read; still a 200 on the route."""
    read = read_device0()
    if read is None or not read.available:
        return {
            "available": False,
            "device_count": 0,
            "name": "",
            "memory_total_bytes": None,
            "memory_used_bytes": None,
        }
    return {
        "available": True,
        "device_count": read.device_count,
        "name": read.name,
        "memory_total_bytes": read.memory_total_bytes,
        "memory_used_bytes": read.memory_used_bytes,
    }


def available_bytes(execution_provider: str) -> int | None:
    """Free bytes on device 0 (total - used), or None when NVML is missing,
    permission is denied, or zero devices — no value is ever guessed."""
    read = read_device0()
    if read is None or not read.available:
        return None
    if read.memory_total_bytes is None or read.memory_used_bytes is None:
        return None
    if execution_provider != "CUDAExecutionProvider":
        return None  # the 2e check only ever runs under CUDA
    return int(read.memory_total_bytes) - int(read.memory_used_bytes)


def evaluate_memory_gate(
    execution_provider: str,
    minima: dict[str, int],
    resident_bytes: int = 0,
) -> dict[str, bool]:
    """SPEC 2e startup gate. minima maps route name -> operator minimum for the
    routes whose check applies (switch true AND CUDA provider). Fixed compare
    order describe, then tag; only operator minima are subtracted. A failed
    NVML read soft-disables every checked route — no value is guessed. Routes
    not in minima are untouched (they load normally).

    resident_bytes is the measured residency delta of role models that were
    already loaded before this read (load-once-before-gate ordering). It is
    added back so the operator-minimum arithmetic compares against the same
    pristine free bytes the spec's gate-before-load wording assumes; only a
    measured value is ever used, never a model-size estimate."""
    result = {name: True for name in minima}
    if execution_provider != "CUDAExecutionProvider" or not minima:
        return result  # CPU provider: the check never runs at all.
    read = read_device0()
    if read is None or not read.available:
        for name in result:
            result[name] = False
        return result
    if read.memory_total_bytes is None or read.memory_used_bytes is None:
        for name in result:
            result[name] = False
        return result
    # Explicit arithmetic: the NVML read was taken with role models resident,
    # so add their measured residency back instead of comparing against a
    # deflated free value; the operator minima below are then subtracted from
    # the reconstructed pristine total, in the fixed describe-then-tag order.
    free = (int(read.memory_total_bytes) - int(read.memory_used_bytes)) + int(resident_bytes)
    for name in ("describe", "tag"):
        if name not in minima:
            continue
        minimum = minima[name]
        if minimum <= free:
            result[name] = True
            free -= minimum  # only the operator minimum, never a model size
        else:
            result[name] = False
    return result
