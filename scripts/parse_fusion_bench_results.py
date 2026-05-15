#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parse benchmark_serving / trtllm-bench JSON outputs into a comparison table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _from_serving(path: Path) -> dict:
    data = _load_json(path)
    p50_e2e = None
    for p, v in data.get("percentiles_e2el_ms", []):
        if int(p) == 50:
            p50_e2e = v
            break
    return {
        "req_per_sec": data.get("request_throughput"),
        "p50_e2e_ms": p50_e2e or data.get("median_e2el_ms"),
        "mean_ttft_ms": data.get("mean_ttft_ms"),
        "mean_tpot_ms": data.get("mean_tpot_ms"),
    }


def _from_trtllm_bench(path: Path) -> dict:
    data = _load_json(path)
    perf = data.get("performance", data)
    return {
        "req_per_sec": perf.get("request_throughput_req_s"),
        "p50_e2e_ms": perf.get("e2e_latency", {}).get("p50"),
        "mean_ttft_ms": perf.get("ttft", {}).get("average"),
        "mean_tpot_ms": perf.get("tpot", {}).get("average"),
    }


def _collect(phase_dir: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for path in sorted(phase_dir.glob("*.json")):
        name = path.stem
        if name.endswith("_trtllm_bench"):
            out[name.replace("_trtllm_bench", "")] = _from_trtllm_bench(path)
        elif name.startswith("con"):
            out[name] = _from_serving(path)
    return out


def _pct_delta(before: float | None, after: float | None) -> str:
    if before is None or after is None or before == 0:
        return "n/a"
    return f"{(after - before) / before * 100:+.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--phase", default="after")
    parser.add_argument("--branch", default="")
    args = parser.parse_args()

    phase_dir = args.result_dir / args.phase
    metrics = _collect(phase_dir)
    summary_path = args.result_dir / f"summary_{args.phase}.json"
    summary_path.write_text(json.dumps(metrics, indent=2))

    before_dir = args.result_dir / "before"
    if before_dir.exists() and args.phase == "after":
        before = _collect(before_dir)
        print("| Scenario | Metric | Before | After | Delta |")
        print("|----------|--------|--------|-------|-------|")
        for scenario in sorted(set(before) | set(metrics)):
            for key, label in [
                ("req_per_sec", "req/sec"),
                ("p50_e2e_ms", "P50 e2e (ms)"),
                ("mean_ttft_ms", "Mean TTFT (ms)"),
                ("mean_tpot_ms", "Mean TPOT (ms)"),
            ]:
                b = before.get(scenario, {}).get(key)
                a = metrics.get(scenario, {}).get(key)
                print(
                    f"| {scenario} | {label} | {b} | {a} | {_pct_delta(b, a)} |"
                )
    else:
        print(json.dumps(metrics, indent=2))

    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
