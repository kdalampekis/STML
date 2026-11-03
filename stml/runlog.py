"""Run logging: every experiment run gets its own folder under runs/.

    runs/2026-10-14_103512_harc_h1_stml/
        config.json      every setting the run used
        environment.json machine, Python, PyTorch, device (mps / cuda / cpu)
        log.txt          everything printed during the run, with timestamps
        metrics.jsonl    one JSON object per measurement (training curves etc.)
        results.json     final numbers
        report.md        a readable summary: question, setup, results table, notes

Nothing is ever overwritten: a new run makes a new folder. Usage:

    with Run("harc_h1_stml", config) as run:
        run.log("starting")
        run.metric({"epoch": 1, "loss": 0.7})
        run.finish(results, report_markdown)
"""

from __future__ import annotations

import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class Run:
    def __init__(self, name: str, config: dict, root: Path | None = None):
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.dir = (root or ROOT / "runs") / f"{stamp}_{name}"
        self.dir.mkdir(parents=True, exist_ok=False)
        self.name, self.config, self.t0 = name, config, time.time()
        (self.dir / "config.json").write_text(json.dumps(config, indent=1, default=str))
        (self.dir / "environment.json").write_text(json.dumps(environment(), indent=1))
        self._log = open(self.dir / "log.txt", "a", buffering=1)
        self._metrics = open(self.dir / "metrics.jsonl", "a", buffering=1)

    def log(self, msg: str) -> None:
        line = f"[{time.time() - self.t0:8.1f}s] {msg}"
        print(line, flush=True)
        self._log.write(line + "\n")

    def metric(self, values: dict) -> None:
        self._metrics.write(json.dumps({"t": round(time.time() - self.t0, 2), **values}, default=float) + "\n")

    def finish(self, results: dict, report: str) -> None:
        (self.dir / "results.json").write_text(json.dumps(results, indent=1, default=float))
        (self.dir / "report.md").write_text(report)
        self.log(f"finished in {(time.time() - self.t0) / 60:.1f} min; report at {self.dir / 'report.md'}")

    def __enter__(self) -> "Run":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc is not None:
            self.log(f"FAILED: {exc_type.__name__}: {exc}")
        self._log.close()
        self._metrics.close()


def environment() -> dict:
    env = {"python": sys.version.split()[0], "platform": platform.platform(), "machine": platform.machine()}
    try:
        import torch
        env["torch"] = torch.__version__
        env["mps_available"] = bool(torch.backends.mps.is_available())
        env["cuda_available"] = bool(torch.cuda.is_available())
    except ImportError:
        env["torch"] = None
    return env


def table(rows: dict[str, dict], cols: list[tuple[str, str, str]]) -> str:
    """Markdown table. cols = [(key, header, format)], e.g. ('solved', 'Solved in 3', '{:.1%}')."""
    head = "| Condition | " + " | ".join(h for _, h, _ in cols) + " |"
    sep = "| --- | " + " | ".join("---:" for _ in cols) + " |"
    lines = [head, sep]
    for name, r in rows.items():
        cells = []
        for k, _, f in cols:
            v = r.get(k)
            cells.append("-" if v is None or v != v else f.format(v))
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
