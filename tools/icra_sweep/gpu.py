"""GPU census, admission control and the single-instance lock.

Exclusivity is enforced by the runner, not by operator discipline: every launch
is admitted only after a census shows the device has room, and every launch and
exit records a census line so an interrupted sweep can be reconstructed from its
journal alone.
"""

from __future__ import annotations

import fcntl
import os
import subprocess
import time
from dataclasses import dataclass

NVIDIA_SMI = "nvidia-smi"


@dataclass(frozen=True)
class ComputeApp:
    pid: int
    process_name: str
    used_mib: int


@dataclass(frozen=True)
class Census:
    apps: tuple[ComputeApp, ...]
    total_mib: int
    used_mib: int
    ok: bool = True
    error: str = ""

    @property
    def total_gb(self) -> float:
        return self.total_mib / 1024.0

    @property
    def used_gb(self) -> float:
        return self.used_mib / 1024.0

    def others(self, own_pids: set[int]) -> tuple[ComputeApp, ...]:
        return tuple(a for a in self.apps if a.pid not in own_pids)

    def used_gb_by(self, apps: tuple[ComputeApp, ...]) -> float:
        return sum(a.used_mib for a in apps) / 1024.0

    def as_dict(self) -> dict:
        return {
            "apps": [{"pid": a.pid, "name": a.process_name, "used_mib": a.used_mib} for a in self.apps],
            "total_mib": self.total_mib,
            "used_mib": self.used_mib,
            "ok": self.ok,
            "error": self.error,
        }

    def line(self) -> str:
        if not self.ok:
            return f"census UNAVAILABLE ({self.error})"
        apps = ", ".join(f"{a.pid}:{a.used_mib}MiB" for a in self.apps) or "none"
        return f"census {len(self.apps)} app(s) [{apps}] {self.used_mib}/{self.total_mib} MiB"


def _smi(args: list[str]) -> str:
    return subprocess.run(
        [NVIDIA_SMI, *args], capture_output=True, text=True, timeout=30, check=True
    ).stdout


def census() -> Census:
    """Read the device's compute apps and memory.

    A census that cannot be taken is reported as not-ok rather than as an empty
    device: "no processes" and "nvidia-smi is missing" must never be confused,
    because the first admits a launch and the second must block one.
    """
    try:
        raw = _smi(
            ["--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader,nounits"]
        )
        apps = []
        for line in raw.splitlines():
            if not line.strip():
                continue
            pid, name, mem = (p.strip() for p in line.split(",", 2))
            apps.append(ComputeApp(int(pid), name, int(float(mem))))
        mem_raw = _smi(["--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"])
        total, used = (int(float(x.strip())) for x in mem_raw.splitlines()[0].split(","))
        return Census(tuple(apps), total, used)
    except Exception as exc:  # nvidia-smi missing, driver wedged, parse change
        return Census((), 0, 0, ok=False, error=f"{type(exc).__name__}: {exc}")


class SweepLock:
    """One sweep per output directory, enforced by flock.

    A second launch in the same directory would otherwise race the first one's
    in-flight run and put two trainings on one GPU.
    """

    def __init__(self, path: str):
        self.path = path
        self._fh = None

    def __enter__(self) -> SweepLock:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._fh = open(self.path, "w")
        try:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self._fh.close()
            self._fh = None
            raise RuntimeError(
                f"another sweep holds {self.path}. Stop it and its children with: "
                f"fuser -k -TERM {self.path}"
            ) from exc
        self._fh.write(f"{os.getpid()}\n")
        self._fh.flush()
        return self

    def __exit__(self, *exc_info) -> None:
        if self._fh is not None:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            self._fh.close()
            self._fh = None


def wait_for_capacity(
    slots: int,
    own_pids: set[int],
    per_run_gb: float,
    headroom_gb: float,
    timeout_s: float,
    poll_s: float = 15.0,
    log=print,
) -> Census:
    """Block until the device can admit one more run, or raise on timeout.

    ``slots`` is the total number of concurrent runs this sweep is allowed. With
    slots == 1 this is strict exclusivity: the device must be empty of compute
    apps, including anyone else's, before a launch is admitted.
    """
    deadline = time.time() + timeout_s
    announced = False
    while True:
        c = census()
        if not c.ok:
            raise RuntimeError(f"GPU census unavailable, refusing to launch: {c.error}")
        others = c.others(own_pids)
        mine = len(c.apps) - len(others)
        if slots <= 1:
            if not c.apps:
                return c
        else:
            free_gb = c.total_gb - c.used_gb - headroom_gb
            if mine < slots and free_gb >= per_run_gb:
                return c
        if not announced:
            log(f"waiting for GPU capacity (slots={slots}) -- {c.line()}")
            announced = True
        if time.time() > deadline:
            raise TimeoutError(f"GPU did not free up within {timeout_s:.0f}s -- {c.line()}")
        time.sleep(poll_s)


def wait_until_clear(own_pids: set[int], timeout_s: float, poll_s: float = 5.0) -> Census:
    """Wait for a finished run's context to actually leave the device.

    A training process that has exited still holds its CUDA context for a moment.
    Launching the next run against that window is how a sweep ends up with two
    live processes on one card.
    """
    deadline = time.time() + timeout_s
    while True:
        c = census()
        if not c.ok or not c.others(own_pids) and not c.apps:
            return c
        if time.time() > deadline:
            return c
        time.sleep(poll_s)
