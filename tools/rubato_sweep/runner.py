"""Execute a sweep: admission control, launch, journal, per-run JSON.

Every rule in here is the fix for a failure this project actually hit.

  * SKIP-IF-COMPLETE. A resumed sweep re-runs only what did not finish.
  * MATCHED-PAIR ORDER. Cells are grouped so a kill lands between complete
    matched sets (see :meth:`SweepCfg.matched_groups`).
  * JOURNAL WITH CENSUS. Start, exit and a GPU compute-app census at every
    boundary, appended to journal.jsonl, so a killed sweep can be reconstructed.
  * STARTUP-ABORT IS FATAL. A run that dies during startup means the stack is
    broken, not that this cell is unlucky. Continuing launched a second process
    against a half-dead device once; now it stops the sweep.
  * NEVER LAUNCH ONTO A BUSY DEVICE. Exclusive mode waits for an empty device,
    and after every run waits for the finished process's context to leave before
    admitting the next.
  * PACKED RUNS ARE TAGGED. A run that shared the device carries
    ``exclusive=false``, so its wall time cannot later be quoted as a timing.
"""

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import threading
import time
from dataclasses import dataclass

from . import gpu, parse
from .config import Cell, SweepCfg


class SweepAbort(RuntimeError):
    """The sweep stopped on a condition that makes further runs meaningless."""


@dataclass
class Launch:
    cell: Cell
    proc: subprocess.Popen
    log_path: str
    telemetry_path: str
    started: float
    peers: int
    exclusive: bool
    peak_mib: int = 0
    _stop: threading.Event | None = None
    _thread: threading.Thread | None = None


class Journal:
    """Append-only record of every boundary, with a census at each."""

    def __init__(self, path: str, echo=print):
        self.path = path
        self.echo = echo
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def write(self, event: str, **fields) -> None:
        rec = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "event": event, **fields}
        with open(self.path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        detail = " ".join(f"{k}={v}" for k, v in fields.items() if k != "census")
        self.echo(f"[{rec['t']}] {event} {detail}")

    def boundary(self, event: str, **fields) -> gpu.Census:
        c = gpu.census()
        self.write(event, census=c.as_dict(), **fields)
        self.echo(f"    {c.line()}")
        return c


def build_command(cfg: SweepCfg, cell: Cell) -> list[str]:
    """The exact training command for one cell.

    The video cadence conversion happens here and only here: the config carries
    an iteration cadence, the CLI wants env steps.
    """
    cmd = [
        os.path.join(cfg.isaaclab_dir, "isaaclab.sh"),
        "train",
        "--rl_library",
        cfg.rl_library,
        "--task",
        cell.task,
        "--num_envs",
        str(cell.num_envs),
        "--seed",
        str(cell.seed),
        "--logger",
        "wandb",
        "--log_project_name",
        cfg.project,
        "--run_name",
        cell.run_name,
        "--run_group",
        f"{cfg.name}-{cell.arm.name}",
    ]
    if cell.iterations is not None:
        cmd += ["--max_iterations", str(cell.iterations)]
    if cfg.viz:
        cmd += ["--viz", cfg.viz]
    if cfg.video.enabled:
        cmd += [
            "--video",
            "--video_length",
            str(cfg.video.length),
            "--video_interval",
            str(cfg.video_interval_env_steps()),
        ]
    cmd += list(cfg.extra_args)
    cmd += cell.arm.cli_args()
    return cmd


def build_env(cfg: SweepCfg, cell: Cell, telemetry_path: str) -> dict[str, str]:
    env = dict(os.environ)
    # isaaclab.sh resolves VIRTUAL_ENV first and ambient shell state varies, so
    # the interpreter is pinned explicitly rather than inherited.
    env["VIRTUAL_ENV"] = cfg.venv_dir
    env["OMNI_KIT_ACCEPT_EULA"] = "YES"
    env["NEWTON_ADAPTIVE_LOG"] = telemetry_path
    env.setdefault("NEWTON_ADAPTIVE_LOG_EVERY", "30")
    env.update(cell.arm.env)
    return env


def _sample_peak(launch: Launch, poll_s: float = 20.0) -> None:
    """Track this run's own peak device memory while it runs."""
    pid = launch.proc.pid
    while not launch._stop.is_set():
        c = gpu.census()
        if c.ok:
            # isaaclab.sh execs a python child, so the compute app's pid is a
            # descendant; attribute by the largest app when we are exclusive.
            mine = [a for a in c.apps if a.pid == pid]
            if not mine and launch.exclusive and c.apps:
                mine = list(c.apps)
            if mine:
                launch.peak_mib = max(launch.peak_mib, max(a.used_mib for a in mine))
        launch._stop.wait(poll_s)


def _finish(cfg: SweepCfg, launch: Launch, journal: Journal) -> parse.RunRecord:
    rc = launch.proc.returncode
    wall = time.time() - launch.started
    if launch._stop is not None:
        launch._stop.set()
    if launch._thread is not None:
        launch._thread.join(timeout=5)
    rec = parse.parse_training_log(launch.log_path)
    rec.run_name = launch.cell.run_name
    rec.arm = launch.cell.arm.name
    rec.task = launch.cell.task
    rec.num_envs = launch.cell.num_envs
    rec.seed = launch.cell.seed
    rec.iterations_requested = launch.cell.iterations
    rec.exit_code = rc
    rec.wall_s = wall
    rec.exclusive = launch.exclusive
    rec.concurrent_peers = launch.peers
    rec.gpu_peak_mib = launch.peak_mib or None
    rec.num_steps_per_env = cfg.num_steps_per_env
    rec.decimation = cfg.decimation
    rec.env_overrides = dict(launch.cell.arm.env)
    rec.stack = stack_fingerprint(cfg)
    if rec.training_time_s is not None and rec.iters:
        iter_total = sum(it.get("iter_time_s", 0.0) for it in rec.iters)
        rec.startup_s = max(0.0, wall - iter_total)
    rec.telemetry = parse.parse_telemetry(launch.telemetry_path)
    accepted = rec.telemetry.get("cumulative_accepted_delta")
    if accepted and rec.iterations_completed:
        wall_iters = sum(it.get("iter_time_s", 0.0) for it in rec.iters)
        rec.demand = parse.normalize_demand(
            accepted_substeps=accepted,
            wall_s=wall_iters,
            iterations=rec.iterations_completed,
            num_steps_per_env=cfg.num_steps_per_env,
            decimation=cfg.decimation,
        )
    if not launch.exclusive:
        rec.notes.append("PACKED: shared the device; wall time is NOT a measurement")
    if rec.first_overflow_iter is not None:
        rec.notes.append(
            f"contact/tri-pair buffer overflowed from iteration {rec.first_overflow_iter}: "
            "physics past that point is a truncated contact set"
        )
    rec.to_json(cfg.json_path(launch.cell))
    journal.boundary(
        "exit",
        run=launch.cell.run_name,
        rc=rc,
        wall_s=round(wall, 1),
        iters=rec.iterations_completed,
        complete=rec.complete,
        packed=not launch.exclusive,
        peak_mib=rec.gpu_peak_mib,
    )
    return rec


def stack_fingerprint(cfg: SweepCfg) -> dict[str, str]:
    """Record which commit of each repo produced these runs.

    sap_warp is joined by sys.path rather than pinned, so a run's physics can
    change with no commit in newton-adaptive or IsaacLab. Recording all three
    is the only way a result stays attributable.
    """
    out: dict[str, str] = {}
    repos = {
        "IsaacLab": cfg.isaaclab_dir,
        "newton-adaptive": os.path.expanduser("~/Documents/code/newton-adaptive"),
        "sap_warp": os.environ.get("SAP_WARP_PATH", os.path.expanduser("~/Documents/code/sap_warp")),
        "IsaacLabRubato": os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    }
    for name, path in repos.items():
        try:
            head = subprocess.run(
                ["git", "-C", path, "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            dirty = subprocess.run(
                ["git", "-C", path, "status", "--porcelain"], capture_output=True, text=True, timeout=30
            )
            if head.returncode == 0:
                out[name] = head.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
        except Exception as exc:
            out[name] = f"unknown ({type(exc).__name__})"
    return out


def run_sweep(cfg: SweepCfg, dry_run: bool = False, echo=print) -> list[parse.RunRecord]:
    """Execute every cell that is not already complete."""
    os.makedirs(cfg.out_dir, exist_ok=True)
    journal = Journal(cfg.journal_path(), echo=echo)
    records: list[parse.RunRecord] = []

    cells = cfg.cells()
    todo = [c for c in cells if not parse.is_complete(cfg.log_path(c), c.iterations)]
    echo(f"sweep '{cfg.name}': {len(cells)} cells, {len(cells) - len(todo)} already complete, {len(todo)} to run")
    if dry_run:
        for c in cells:
            state = "COMPLETE" if c not in todo else "todo"
            echo(f"  [{state:8}] {c.run_name}  {shlex.join(build_command(cfg, c))}")
        return records

    journal.write(
        "sweep_start",
        name=cfg.name,
        config=cfg.source,
        cells=len(cells),
        todo=len(todo),
        timing_sensitive=cfg.timing_sensitive,
        packing=cfg.packing.enabled,
        stack=stack_fingerprint(cfg),
    )

    slots = 1
    if cfg.packing.enabled:
        c0 = gpu.census()
        if not c0.ok:
            raise SweepAbort(f"cannot size packing without a census: {c0.error}")
        slots = cfg.packing.slots(c0.total_gb, c0.used_gb)
        echo(f"packing enabled: {slots} concurrent slot(s) on a {c0.total_gb:.1f} GB device")

    live: list[Launch] = []
    try:
        for cell in cells:
            if parse.is_complete(cfg.log_path(cell), cell.iterations):
                journal.write("skip_complete", run=cell.run_name)
                continue
            # Reap finished runs before admitting another.
            live = _reap(cfg, live, journal, records, blocking=len(live) >= slots)
            own = {lc.proc.pid for lc in live}
            gpu.wait_for_capacity(
                slots=slots,
                own_pids=own,
                per_run_gb=cfg.packing.per_run_gb,
                headroom_gb=cfg.packing.headroom_gb,
                timeout_s=cfg.timeout_s,
                log=echo,
            )
            live.append(_launch(cfg, cell, journal, peers=len(live), echo=echo))
            if slots == 1:
                live = _reap(cfg, live, journal, records, blocking=True)
                # Let the finished process's CUDA context actually go away
                # before the next launch is admitted.
                gpu.wait_until_clear(own_pids=set(), timeout_s=180)
        while live:
            live = _reap(cfg, live, journal, records, blocking=True)
    finally:
        for lc in live:
            if lc.proc.poll() is None:
                lc.proc.send_signal(signal.SIGTERM)
        journal.write("sweep_end", completed=len(records))
    return records


def _launch(cfg: SweepCfg, cell: Cell, journal: Journal, peers: int, echo) -> Launch:
    log_path = cfg.log_path(cell)
    telemetry_path = os.path.join(cfg.out_dir, f"{cell.run_name}.telemetry")
    cmd = build_command(cfg, cell)
    env = build_env(cfg, cell, telemetry_path)
    journal.boundary(
        "start",
        run=cell.run_name,
        arm=cell.arm.name,
        num_envs=cell.num_envs,
        seed=cell.seed,
        iterations=cell.iterations,
        peers=peers,
        cmd=shlex.join(cmd),
    )
    log = open(log_path, "a")
    log.write(f"\n==== {time.strftime('%F %T')} launch {cell.run_name} :: {shlex.join(cmd)} ====\n")
    log.flush()
    proc = subprocess.Popen(
        ["timeout", "--signal=TERM", "--kill-after=60", str(cfg.timeout_s), *cmd],
        cwd=cfg.isaaclab_dir,  # the log dir is resolved relative to CWD
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    launch = Launch(
        cell=cell,
        proc=proc,
        log_path=log_path,
        telemetry_path=telemetry_path,
        started=time.time(),
        peers=peers,
        exclusive=peers == 0 and not cfg.packing.enabled,
    )
    launch._stop = threading.Event()
    launch._thread = threading.Thread(target=_sample_peak, args=(launch,), daemon=True)
    launch._thread.start()
    return launch


def _reap(
    cfg: SweepCfg, live: list[Launch], journal: Journal, records: list[parse.RunRecord], blocking: bool
) -> list[Launch]:
    """Collect finished runs; enforce the abort conditions."""
    while True:
        still: list[Launch] = []
        for lc in live:
            if lc.proc.poll() is None:
                still.append(lc)
                continue
            rec = _finish(cfg, lc, journal)
            records.append(rec)
            _check_fatal(cfg, lc, rec, journal)
        if not blocking or len(still) < len(live) or not still:
            return still
        live = still
        time.sleep(10)


def _check_fatal(cfg: SweepCfg, launch: Launch, rec: parse.RunRecord, journal: Journal) -> None:
    """Decide whether this run's outcome invalidates the rest of the sweep."""
    rc = rec.exit_code
    elapsed = rec.wall_s or 0.0
    if rec.oom:
        journal.write("oom", run=rec.run_name, packed=not launch.exclusive)
        if not launch.exclusive:
            # Packing was too aggressive for this scene; fall back rather than
            # burn the rest of the sweep on the same mistake.
            cfg.packing = type(cfg.packing)(
                enabled=False,
                per_run_gb=cfg.packing.per_run_gb,
                headroom_gb=cfg.packing.headroom_gb,
                max_concurrent=cfg.packing.max_concurrent,
            )
            journal.write("packing_disabled", reason="OOM under packing; continuing exclusively")
            return
        raise SweepAbort(f"{rec.run_name} ran out of device memory at {rec.num_envs} envs")
    if rc != 0 and rec.iterations_completed == 0 and elapsed < cfg.startup_abort_s:
        raise SweepAbort(
            f"{rec.run_name} exited rc={rc} after {elapsed:.0f}s without reaching iteration 0. "
            "That is a broken stack, not an unlucky cell -- fix it before spending more GPU time. "
            f"See {rec.log_path}"
        )
    if rc == 124:
        journal.write("timeout", run=rec.run_name, timeout_s=cfg.timeout_s)
