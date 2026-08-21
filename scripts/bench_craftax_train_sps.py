#!/usr/bin/env python3
"""Sweep training-throughput (SPS) configs for craftax and craftax_clean.

Keeps policy architecture and learning-rate/PPO coefficients from the env
ini, and searches the knobs that actually move wall-clock agent steps:
vec.total_agents, vec.num_buffers, vec.num_threads, train.horizon, and
env.reset_pool_size.

Each trial runs a short native `./puffer train` loop and records median
SPS from the run log under logs/<env>/<run_id>.ini.

Examples:
    python3 scripts/bench_craftax_train_sps.py
    python3 scripts/bench_craftax_train_sps.py --env craftax --write-ini
    python3 scripts/bench_craftax_train_sps.py --env craftax_clean --quick
"""

from __future__ import annotations

import argparse
import ast
import configparser
import json
import os
import re
import statistics
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DEFAULT_INI = REPO / "config" / "default.ini"
PUFFER_BIN = REPO / "puffer"


def load_ini(env_name: str) -> dict:
    path = REPO / "config" / f"{env_name}.ini"
    parser = configparser.ConfigParser()
    parser.read([DEFAULT_INI, path])
    args: dict = {}
    for section in parser.sections():
        section_dict = {}
        for key in parser[section]:
            raw = parser[section][key]
            try:
                value = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                value = raw
            section_dict[key] = value
        if section == "base":
            args.update(section_dict)
        else:
            args[section] = section_dict
    args["env_name"] = env_name
    args.setdefault("env", {})
    args.setdefault("vec", {})
    args.setdefault("train", {})
    args.setdefault("policy", {})
    args.setdefault("sweep", {})
    return args


def apply_overrides(args: dict, overrides: dict) -> dict:
    out = deepcopy(args)
    for dotted, value in overrides.items():
        section, key = dotted.split(".", 1)
        if section not in out:
            out[section] = {}
        out[section][key] = value
    return out


def config_ok(args: dict) -> str | None:
    minibatch = int(args["train"]["minibatch_size"])
    horizon = int(args["train"]["horizon"])
    agents = int(args["vec"]["total_agents"])
    buffers = int(args["vec"]["num_buffers"])
    threads = int(args["vec"]["num_threads"])
    if horizon <= 0 or agents <= 0 or buffers <= 0 or threads <= 0:
        return "non-positive layout value"
    if agents % buffers != 0:
        return f"total_agents {agents} not divisible by num_buffers {buffers}"
    if minibatch % horizon != 0:
        return f"minibatch_size {minibatch} not divisible by horizon {horizon}"
    if minibatch > horizon * agents:
        return (
            f"minibatch_size {minibatch} > total_agents {agents} * horizon {horizon}"
        )
    rows = minibatch // horizon
    if agents % rows != 0:
        return f"total_agents {agents} not divisible by minibatch rows {rows}"
    if horizon % 4 != 0:
        return f"horizon {horizon} is not a multiple of ADV_VEC_WIDTH"
    return None


def parse_sps_token(token: str) -> float:
    token = token.strip().replace(",", "")
    multiplier = 1.0
    if token.endswith(("K", "k")):
        multiplier = 1e3
        token = token[:-1]
    elif token.endswith(("M", "m")):
        multiplier = 1e6
        token = token[:-1]
    elif token.endswith(("B", "b")):
        multiplier = 1e9
        token = token[:-1]
    return float(token) * multiplier


def parse_dashboard_sps(text: str) -> list[float]:
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    samples = []
    for match in re.finditer(r"SPS(?:\s+|=)([0-9]+(?:\.[0-9]+)?)([KMB]?)", text):
        samples.append(parse_sps_token(match.group(1) + match.group(2)))
    return samples


def parse_metric_series(raw: str) -> list[float]:
    values = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    return values


def read_run_sps(log_path: Path) -> list[float]:
    parser = configparser.ConfigParser()
    parser.optionxform = str
    parser.read(log_path)
    if not parser.has_section("metrics") or not parser.has_option("metrics", "SPS"):
        raise RuntimeError(f"no [metrics] SPS in {log_path}")
    samples = parse_metric_series(parser["metrics"]["SPS"])
    if not samples:
        raise RuntimeError(f"empty SPS series in {log_path}")
    return samples


def summarize_sps(samples: list[float], warmup_epochs: int) -> dict:
    stable = list(samples)
    if len(stable) >= 2 and stable[-1] > 2.0 * stable[-2]:
        stable = stable[:-1]
    measured = stable[warmup_epochs:] if len(stable) > warmup_epochs else stable[-1:]
    if not measured:
        measured = stable[-1:]
    return {
        "median_sps": statistics.median(measured),
        "mean_sps": statistics.mean(measured),
        "min_sps": min(measured),
        "max_sps": max(measured),
        "samples": samples,
        "measured_samples": measured,
    }


def run_trial(
    env_name: str,
    cfg: dict,
    warmup_epochs: int,
    measure_epochs: int,
    cudagraphs: int,
    trial_id: str,
) -> dict:
    agents = int(cfg["vec"]["total_agents"])
    horizon = int(cfg["train"]["horizon"])
    epochs = warmup_epochs + measure_epochs
    timesteps = max(agents * horizon * epochs, agents * horizon)
    downsample = min(64, max(1, epochs))
    run_id = f"sps_{env_name}_{trial_id}"
    log_path = REPO / "logs" / env_name / f"{run_id}.ini"

    cmd = [
        str(PUFFER_BIN),
        "train",
        f"--base.run_id={run_id}",
        f"--base.cudagraphs={int(cudagraphs)}",
        "--base.eval_episodes=0",
        "--base.checkpoint_interval=1000000",
        f"--sweep.downsample={downsample}",
        f"--train.total_timesteps={timesteps}",
        f"--vec.total_agents={agents}",
        f"--vec.num_buffers={int(cfg['vec']['num_buffers'])}",
        f"--vec.num_threads={int(cfg['vec']['num_threads'])}",
        f"--train.horizon={horizon}",
        f"--train.minibatch_size={int(cfg['train']['minibatch_size'])}",
        f"--env.reset_pool_size={int(cfg['env'].get('reset_pool_size', 0) or 0)}",
    ]

    t_init = time.time()
    proc = subprocess.run(
        cmd,
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - t_init
    output = (proc.stdout or "") + (proc.stderr or "")
    if output:
        print(output, end="" if output.endswith("\n") else "\n", flush=True)

    samples = []
    if proc.returncode == 0 and log_path.exists():
        try:
            samples = read_run_sps(log_path)
        except RuntimeError:
            samples = []
    if not samples:
        samples = parse_dashboard_sps(output)
    missing_env_perf = "missing key" in output and "env/perf" in output
    if proc.returncode != 0 and not (samples and missing_env_perf):
        tail = output.strip().splitlines()[-20:]
        detail = "\n".join(tail) if tail else f"exit {proc.returncode}"
        raise RuntimeError(f"./puffer train failed ({proc.returncode}): {detail}")
    if not samples:
        raise RuntimeError(f"no SPS samples for {run_id}")

    result = summarize_sps(samples, warmup_epochs)
    result["init_s"] = elapsed
    result["run_id"] = run_id
    result["log_path"] = str(log_path) if log_path.exists() else None
    result["timesteps"] = timesteps
    return result


def build_env(env_name: str) -> None:
    print(f"Building {env_name} ...", flush=True)
    env = os.environ.copy()
    python = env.get("PYTHON") or sys.executable
    env["PATH"] = f"{Path(python).parent}:{env.get('PATH', '')}"
    env["PYTHON"] = python
    subprocess.run(
        ["./build.sh", env_name],
        cwd=REPO,
        check=True,
        env=env,
    )
    if not PUFFER_BIN.exists():
        raise RuntimeError(f"{PUFFER_BIN} was not produced by ./build.sh {env_name}")


def search_space(env_name: str, quick: bool) -> list[dict]:
    """Coordinate search around the current ini, then a small 2D refine."""
    if quick:
        agents = [8192, 16384]
        buffers = [8, 16]
        threads = [16]
        pools = [32]
        horizons = [128]
    else:
        agents = [8192, 16384, 32768]
        buffers = [4, 8, 16]
        threads = [8, 16]
        pools = [32, 128, 256]
        horizons = [64, 128]

    configs: list[dict] = []
    seen: set[tuple] = set()

    def add(**kwargs):
        key = tuple(sorted(kwargs.items()))
        if key in seen:
            return
        seen.add(key)
        configs.append(kwargs)

    add()

    for pool in pools:
        add(**{"env.reset_pool_size": pool})
    for n_threads in threads:
        add(**{"vec.num_threads": n_threads, "env.reset_pool_size": pools[0]})
    for n_buffers in buffers:
        add(
            **{
                "vec.num_buffers": n_buffers,
                "vec.num_threads": 16,
                "env.reset_pool_size": pools[0],
            }
        )
    for n_agents in agents:
        add(
            **{
                "vec.total_agents": n_agents,
                "vec.num_buffers": 16,
                "vec.num_threads": 16,
                "env.reset_pool_size": pools[0],
            }
        )
    for horizon in horizons:
        add(
            **{
                "vec.total_agents": 16384,
                "vec.num_buffers": 16,
                "vec.num_threads": 16,
                "train.horizon": horizon,
                "env.reset_pool_size": pools[0],
            }
        )

    for n_agents in agents:
        for n_buffers in buffers:
            add(
                **{
                    "vec.total_agents": n_agents,
                    "vec.num_buffers": n_buffers,
                    "vec.num_threads": 16,
                    "env.reset_pool_size": pools[-1] if env_name == "craftax" else pools[0],
                }
            )

    if not quick:
        add(**{"env.reset_pool_size": 0, "vec.total_agents": 4096, "vec.num_buffers": 8})
        add(
            **{
                "vec.total_agents": 65536,
                "vec.num_buffers": 16,
                "vec.num_threads": 16,
                "env.reset_pool_size": pools[-1],
            }
        )
        add(
            **{
                "vec.total_agents": 32768,
                "vec.num_buffers": 8,
                "vec.num_threads": 16,
                "train.horizon": 64,
                "env.reset_pool_size": pools[-1],
            }
        )
    return configs


def format_sps(value: float) -> str:
    if value >= 1e6:
        return f"{value / 1e6:.2f}M"
    if value >= 1e3:
        return f"{value / 1e3:.1f}K"
    return f"{value:.0f}"


def run_env_search(env_name: str, args: argparse.Namespace) -> dict:
    base = load_ini(env_name)
    trials = []
    best = None
    for index, overrides in enumerate(search_space(env_name, args.quick)):
        if args.max_trials is not None and len(trials) >= args.max_trials:
            break
        cfg = apply_overrides(base, overrides)
        reason = config_ok(cfg)
        label = overrides or {"(ini baseline)": True}
        if reason:
            print(f"SKIP {env_name} {label}: {reason}", flush=True)
            continue
        print(f"TRIAL {env_name} {label}", flush=True)
        t0 = time.time()
        try:
            result = run_trial(
                env_name,
                cfg,
                warmup_epochs=args.warmup_epochs,
                measure_epochs=args.measure_epochs,
                cudagraphs=args.cudagraphs,
                trial_id=f"{int(time.time() * 1000)}_{index}",
            )
        except Exception as exc:  # noqa: BLE001 - trial isolation
            elapsed = time.time() - t0
            print(f"  FAIL after {elapsed:.1f}s: {exc}", flush=True)
            trials.append(
                {
                    "overrides": overrides,
                    "error": str(exc),
                    "elapsed_s": elapsed,
                }
            )
            continue
        elapsed = time.time() - t0
        result["overrides"] = overrides
        result["elapsed_s"] = elapsed
        result["config"] = {
            "vec.total_agents": int(cfg["vec"]["total_agents"]),
            "vec.num_buffers": int(cfg["vec"]["num_buffers"]),
            "vec.num_threads": int(cfg["vec"]["num_threads"]),
            "train.horizon": int(cfg["train"]["horizon"]),
            "train.minibatch_size": int(cfg["train"]["minibatch_size"]),
            "env.reset_pool_size": int(cfg["env"].get("reset_pool_size", 0) or 0),
        }
        trials.append(result)
        print(
            f"  SPS median={format_sps(result['median_sps'])} "
            f"mean={format_sps(result['mean_sps'])} "
            f"range={format_sps(result['min_sps'])}-{format_sps(result['max_sps'])} "
            f"trial={elapsed:.1f}s run_id={result['run_id']}",
            flush=True,
        )
        if best is None or result["median_sps"] > best["median_sps"]:
            best = result

    if best is None:
        raise RuntimeError(f"no successful trials for {env_name}")
    return {"env": env_name, "best": best, "trials": trials}


def write_ini(env_name: str, best_config: dict) -> None:
    path = REPO / "config" / f"{env_name}.ini"
    text = path.read_text()

    def replace_key(section_header: str, key: str, value: int, body: str) -> str:
        lines = body.splitlines(keepends=True)
        out = []
        in_section = False
        replaced = False
        header = f"[{section_header}]"
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                in_section = stripped == header
            if in_section and stripped.startswith(f"{key} "):
                indent = line[: len(line) - len(line.lstrip())]
                out.append(f"{indent}{key} = {value}\n")
                replaced = True
                continue
            if in_section and stripped.startswith(f"{key}="):
                indent = line[: len(line) - len(line.lstrip())]
                out.append(f"{indent}{key} = {value}\n")
                replaced = True
                continue
            out.append(line)
        if not replaced:
            raise RuntimeError(f"could not find {section_header}.{key} in {path}")
        return "".join(out)

    mapping = {
        ("vec", "total_agents"): best_config["vec.total_agents"],
        ("vec", "num_buffers"): best_config["vec.num_buffers"],
        ("vec", "num_threads"): best_config["vec.num_threads"],
        ("train", "horizon"): best_config["train.horizon"],
        ("env", "reset_pool_size"): best_config["env.reset_pool_size"],
    }
    for (section, key), value in mapping.items():
        text = replace_key(section, key, value, text)
    path.write_text(text)
    print(f"Updated {path}")


def print_peak(reports: list[dict]) -> None:
    print("\n=== Peak training SPS ===")
    for report in reports:
        best = report["best"]
        cfg = best["config"]
        print(
            f"{report['env']}: {format_sps(best['median_sps'])} median SPS "
            f"(mean {format_sps(best['mean_sps'])})"
        )
        print(
            "  "
            f"total_agents={cfg['vec.total_agents']} "
            f"num_buffers={cfg['vec.num_buffers']} "
            f"num_threads={cfg['vec.num_threads']} "
            f"horizon={cfg['train.horizon']} "
            f"minibatch_size={cfg['train.minibatch_size']} "
            f"reset_pool_size={cfg['env.reset_pool_size']}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env",
        choices=["craftax", "craftax_clean", "both"],
        default="both",
    )
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--measure-epochs", type=int, default=5)
    parser.add_argument("--cudagraphs", type=int, default=1)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--max-trials",
        type=int,
        default=None,
        help="Stop after this many attempted trials per env (skips do not count)",
    )
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--write-ini", action="store_true")
    parser.add_argument(
        "--results",
        type=Path,
        default=REPO / "logs" / "craftax_train_sps.json",
    )
    cli = parser.parse_args()

    os.chdir(REPO)
    env_names = ["craftax", "craftax_clean"] if cli.env == "both" else [cli.env]
    reports = []
    for env_name in env_names:
        if not cli.skip_build:
            build_env(env_name)
        elif not PUFFER_BIN.exists():
            raise RuntimeError(f"{PUFFER_BIN} missing; rebuild or drop --skip-build")
        reports.append(run_env_search(env_name, cli))

    cli.results.parent.mkdir(parents=True, exist_ok=True)
    cli.results.write_text(json.dumps(reports, indent=2))
    print_peak(reports)
    if cli.write_ini:
        for report in reports:
            write_ini(report["env"], report["best"]["config"])
    print(f"\nWrote {cli.results}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
