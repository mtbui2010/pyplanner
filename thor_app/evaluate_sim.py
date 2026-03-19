"""
evaluate_sim.py
===============
Execute generated plans inside AI2-THOR via ThorClient and measure
ground-truth execution metrics.

Requires:
  - python thor_server.py  (running in a separate terminal)
  - python make_dataset.py  (to generate eval_dataset.json first)

Metrics added on top of evaluate.py's static metrics:
  Execution:
    exec_task_success   — 1 if all steps succeeded OR env signals done
    exec_step_success   — fraction of steps that returned success=True
    exec_total_reward   — cumulative reward from simulator
    exec_steps_done     — steps actually executed (may stop early on fail)
    exec_replans        — number of replan calls triggered during execution
    exec_latency_s      — wall-clock time for the full execute loop

  Combined (static + execution):
    combined_score      — 0.5 × quality_score + 0.5 × exec_task_success

Usage:
    # All methods, default dataset:
    python evaluate_sim.py

    # Select methods / model / output:
    python evaluate_sim.py --methods Direct CoT --model llama3.2 --out sim_results.csv

    # Limit samples (faster for debugging):
    python evaluate_sim.py --max-samples 5

    # Dry-run LLM (still executes in simulator):
    python evaluate_sim.py --dry-run-llm

    # Skip simulator execution, combine with offline metrics only:
    python evaluate_sim.py --no-sim
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from typing import Any

# ── Resolve paths ─────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
try:
    import pyplanner
except ModuleNotFoundError:
    sys.path.insert(0, os.path.join(_HERE, "..", "pyplanner"))
    import pyplanner

from pyplanner import REGISTRY, DEFAULT_HOST, DEFAULT_MODEL, DEFAULT_BACKEND

try:
    from thor_client import ThorClient
except ImportError:
    sys.path.insert(0, _HERE)
    from thor_client import ThorClient

try:
    from make_dataset import (
        build_dataset, validate_dataset,
        score_executability, score_precondition, score_redundancy,
        score_completeness, score_hallucination, compute_quality_score,
    )
    from evaluate import (
        check_connection, compute_efficiency_score,
        compute_robustness_score, compute_overall_score,
        _DryRunPlanner, CSV_COLUMNS as OFFLINE_COLUMNS,
        SampleResult as OfflineSampleResult,
    )
except ImportError:
    sys.path.insert(0, _HERE)
    from make_dataset import (
        build_dataset, validate_dataset,
        score_executability, score_precondition, score_redundancy,
        score_completeness, score_hallucination, compute_quality_score,
    )
    from evaluate import (
        check_connection, compute_efficiency_score,
        compute_robustness_score, compute_overall_score,
        _DryRunPlanner, CSV_COLUMNS as OFFLINE_COLUMNS,
        SampleResult as OfflineSampleResult,
    )


# ═════════════════════════════════════════════════════════════════════
# Result dataclass — extends offline result with execution columns
# ═════════════════════════════════════════════════════════════════════

SIM_EXTRA_COLUMNS = [
    "exec_task_success",
    "exec_step_success",
    "exec_total_reward",
    "exec_steps_done",
    "exec_replans",
    "exec_latency_s",
    "sim_error",
    "combined_score",
]

SIM_CSV_COLUMNS = OFFLINE_COLUMNS + SIM_EXTRA_COLUMNS


@dataclass
class SimSampleResult:
    # ── Copy of offline fields ──
    method:     str   = ""
    model:      str   = ""
    backend:    str   = ""
    task_id:    str   = ""
    task_desc:  str   = ""
    room:       str   = ""
    difficulty: str   = ""

    parse_ok:        float = 0.0
    num_steps:       int   = 0
    ref_steps:       int   = 0
    step_ratio:      float = 0.0
    executability:   float = 0.0
    precondition:    float = 0.0
    redundancy:      float = 0.0
    completeness:    float = 0.0
    hallucination:   float = 0.0
    quality_score:   float = 0.0

    latency_s:        float = 0.0
    llm_calls:        int   = 0
    input_tokens:     int   = 0
    output_tokens:    int   = 0
    total_tokens:     int   = 0
    tokens_per_step:  float = 0.0
    efficiency_score: float = 0.0

    has_fail_injection: int   = 0
    replan_ok:          float = 0.0
    replan_latency_s:   float = 0.0
    replan_steps:       int   = 0
    replan_llm_calls:   int   = 0
    step_overlap:       float = 0.0
    robustness_score:   float = 0.0

    overall_score: float = 0.0
    error:         str   = ""

    # ── Execution-specific fields ──
    exec_task_success: float = 0.0   # 1.0 if task completed successfully
    exec_step_success: float = 0.0   # fraction of steps that returned success
    exec_total_reward: float = 0.0   # cumulative reward from simulator
    exec_steps_done:   int   = 0     # how many steps were actually executed
    exec_replans:      int   = 0     # replan calls triggered
    exec_latency_s:    float = 0.0   # wall time for execution loop
    sim_error:         str   = ""    # simulator-level error
    combined_score:    float = 0.0   # 0.5×quality + 0.5×exec_task_success

    def to_row(self) -> dict:
        return asdict(self)


# ═════════════════════════════════════════════════════════════════════
# Simulator connection check
# ═════════════════════════════════════════════════════════════════════

def check_simulator(host: str, port: int) -> tuple[bool, str]:
    """Ping ThorServer and return (ok, message)."""
    try:
        client = ThorClient(host=host, port=port)
        if client.connected:
            return True, f"ThorServer OK at {host}:{port}"
        return False, (
            f"ThorServer not responding at {host}:{port}\n"
            f"  Fix: run  python thor_server.py  in a separate terminal"
        )
    except Exception as e:
        return False, (
            f"Cannot connect to ThorServer at {host}:{port}\n"
            f"  Error: {e}\n"
            f"  Fix: run  python thor_server.py  in a separate terminal"
        )


# ═════════════════════════════════════════════════════════════════════
# Core: execute plan in simulator
# ═════════════════════════════════════════════════════════════════════

def execute_plan_in_sim(
    client:    ThorClient,
    planner,
    sample:    dict,
    plan:      list[dict],
    max_replan: int = 3,
    verbose:   bool = False,
) -> dict:
    """
    Execute a generated plan step-by-step in AI2-THOR.

    Returns a dict with execution metrics:
        task_success, step_success, total_reward,
        steps_done, replans, latency_s, error
    """
    t0               = time.perf_counter()
    completed        = []
    current_plan     = list(plan)
    replan_count     = 0
    steps_attempted  = 0
    steps_succeeded  = 0
    total_reward     = 0.0
    task_done        = False
    sim_error        = ""

    try:
        # Reset simulator to the correct scene
        scene = sample.get("scene", "FloorPlan1")
        resp  = client.reset(scene)
        if resp.get("status") != "ok":
            return {
                "task_success": 0.0, "step_success": 0.0,
                "total_reward": 0.0, "steps_done": 0,
                "replans": 0, "latency_s": 0.0,
                "error": f"Scene reset failed: {resp.get('msg','')}",
            }

        obs             = resp.get("obs", "")
        visible_objects = resp.get("visible_objects", [])

        # Execute steps
        while current_plan:
            step   = current_plan.pop(0)
            action = step.get("action", "Wait")
            obj    = step.get("object", "")
            target = step.get("target", "")

            if verbose:
                print(f"    → {action} {obj}" + (f" → {target}" if target else ""))

            result = client.step(action, obj, target)
            steps_attempted += 1

            obs             = result.get("obs", obs)
            visible_objects = result.get("visible_objects", visible_objects)
            reward          = result.get("reward", 0.0)
            success         = result.get("success", False)
            total_reward   += reward

            if result.get("done"):
                task_done = True
                steps_succeeded += 1
                break

            if success:
                completed.append(step)
                steps_succeeded += 1
            else:
                # Step failed — try replan
                if replan_count < max_replan and planner is not None:
                    replan_count += 1
                    msg = result.get("msg", "step failed")
                    if verbose:
                        print(f"    ⚠  Failed: {msg} — replanning ({replan_count}/{max_replan})")
                    try:
                        new_steps, _ = planner.replan(
                            task            = sample["task_desc"],
                            completed       = completed,
                            failed_step     = step,
                            failure_reason  = msg,
                            obs             = obs,
                            visible_objects = visible_objects,
                        )
                        current_plan = new_steps
                    except Exception as e:
                        if verbose:
                            print(f"    ✖  Replan error: {e}")
                        break
                else:
                    break  # max replans reached or no planner

        # If we exhausted all steps without task_done, check if reward > 0
        if not task_done and total_reward > 0:
            task_done = True

    except Exception as e:
        sim_error = str(e)[:150]
        if verbose:
            traceback.print_exc()

    elapsed = time.perf_counter() - t0
    step_success_rate = round(steps_succeeded / steps_attempted, 4) if steps_attempted else 0.0

    return {
        "task_success": 1.0 if task_done else 0.0,
        "step_success": step_success_rate,
        "total_reward": round(total_reward, 4),
        "steps_done":   steps_attempted,
        "replans":      replan_count,
        "latency_s":    round(elapsed, 3),
        "error":        sim_error,
    }


# ═════════════════════════════════════════════════════════════════════
# Evaluate one sample: plan + (optionally) execute
# ═════════════════════════════════════════════════════════════════════

def evaluate_sample_sim(
    client:     ThorClient | None,
    planner,
    sample:     dict,
    max_replan: int  = 3,
    run_sim:    bool = True,
    verbose:    bool = False,
) -> SimSampleResult:

    res = SimSampleResult(
        method     = planner.name,
        model      = getattr(planner, "model", ""),
        backend    = getattr(planner, "provider", ""),
        task_id    = sample["task_id"],
        task_desc  = sample["task_desc"][:60],
        room       = sample["room"],
        difficulty = sample["difficulty"],
        ref_steps  = len(sample["reference_steps"]),
        has_fail_injection = 1 if sample.get("fail_injection") else 0,
    )

    plan = []
    try:
        # ── Phase 1: Generate plan ──────────────────────────────────────
        steps, metrics = planner.generate_plan(
            task            = sample["task_desc"],
            obs             = sample["obs"],
            visible_objects = sample["visible_objects"],
        )

        if metrics.notes and not metrics.parse_ok:
            raise RuntimeError(f"LLM error: {metrics.notes}")

        plan = steps

        res.parse_ok        = 1.0 if (steps and metrics.parse_ok) else 0.0
        res.num_steps       = len(steps)
        res.step_ratio      = round(res.num_steps / res.ref_steps, 4) if res.ref_steps else 0.0
        res.latency_s       = round(metrics.latency_s, 4)
        res.llm_calls       = metrics.llm_calls
        res.input_tokens    = metrics.input_tokens
        res.output_tokens   = metrics.output_tokens
        res.total_tokens    = metrics.total_tokens
        res.tokens_per_step = round(metrics.tokens_per_step, 2)

        res.executability = score_executability(steps)
        res.precondition  = score_precondition(steps)
        res.redundancy    = score_redundancy(steps)
        res.completeness  = score_completeness(steps, sample["expected_objects"])
        res.hallucination = score_hallucination(steps, sample["visible_objects"])
        res.quality_score = compute_quality_score({
            "executability": res.executability,
            "precondition":  res.precondition,
            "completeness":  res.completeness,
            "redundancy":    res.redundancy,
            "hallucination": res.hallucination,
        })
        res.efficiency_score = compute_efficiency_score(res.latency_s, res.llm_calls)
        res.overall_score    = compute_overall_score(
            res.quality_score, res.efficiency_score, 0.0, False
        )

    except Exception as e:
        res.error = str(e)[:200]
        print(f"\n  ⚠  [{res.task_id}] {res.method} plan error: {str(e)[:80]}")
        return res  # skip execution if planning failed

    if verbose:
        print(f"  📋 [{res.task_id}] {len(plan)} steps generated "
              f"(Q={res.quality_score:.2f}  {res.latency_s:.1f}s)")

    # ── Phase 2: Execute in simulator ──────────────────────────────────
    if run_sim and client is not None and plan:
        exec_result = execute_plan_in_sim(
            client    = client,
            planner   = planner,
            sample    = sample,
            plan      = plan,
            max_replan= max_replan,
            verbose   = verbose,
        )
        res.exec_task_success = exec_result["task_success"]
        res.exec_step_success = exec_result["step_success"]
        res.exec_total_reward = exec_result["total_reward"]
        res.exec_steps_done   = exec_result["steps_done"]
        res.exec_replans      = exec_result["replans"]
        res.exec_latency_s    = exec_result["latency_s"]
        res.sim_error         = exec_result["error"]

        if res.sim_error:
            print(f"\n  ⚠  [{res.task_id}] sim error: {res.sim_error[:80]}")

        if verbose:
            status = "✅" if res.exec_task_success else "❌"
            print(f"  {status} [{res.task_id}] exec: "
                  f"success={res.exec_task_success}  "
                  f"steps={res.exec_steps_done}/{res.num_steps}  "
                  f"reward={res.exec_total_reward:.1f}  "
                  f"{res.exec_latency_s:.1f}s")

    # Combined score: blend plan quality with execution success
    if run_sim:
        res.combined_score = round(
            0.5 * res.quality_score + 0.5 * res.exec_task_success, 4
        )
    else:
        res.combined_score = res.quality_score

    return res


# ═════════════════════════════════════════════════════════════════════
# Summary printer
# ═════════════════════════════════════════════════════════════════════

def _print_summary(results: list[SimSampleResult], run_sim: bool):
    from collections import defaultdict
    by_method: dict[str, list[SimSampleResult]] = defaultdict(list)
    for r in results:
        by_method[r.method].append(r)

    def avg(rows, attr):
        vals = [getattr(r, attr) for r in rows]
        return round(sum(vals) / len(vals), 3) if vals else 0.0

    if run_sim:
        header = (
            f"\n{'─'*130}\n"
            f"  {'Method':<16} {'Model':<14} "
            f" {'Quality':>7} {'Effic':>6} {'Overall':>8}"
            f" {'TaskSucc':>8} {'StepSucc':>8} {'Reward':>7} {'ExecLat':>8}"
            f" {'Combined':>9} {'Errors':>7}\n"
            f"{'─'*130}"
        )
        print(header)
        for method, rows in by_method.items():
            n   = len(rows)
            err = sum(1 for r in rows if r.error or r.sim_error)
            print(
                f"  {method:<16} {rows[0].model:<14}"
                f" {avg(rows,'quality_score'):>7.3f}"
                f" {avg(rows,'efficiency_score'):>6.3f}"
                f" {avg(rows,'overall_score'):>8.3f}"
                f" {avg(rows,'exec_task_success'):>8.3f}"
                f" {avg(rows,'exec_step_success'):>8.3f}"
                f" {avg(rows,'exec_total_reward'):>7.2f}"
                f" {avg(rows,'exec_latency_s'):>7.1f}s"
                f" {avg(rows,'combined_score'):>9.3f}"
                f" {err:>5}/{n}"
            )
    else:
        header = (
            f"\n{'─'*110}\n"
            f"  {'Method':<16} {'Model':<14}"
            f" {'Quality':>7} {'Effic':>6} {'Overall':>8}"
            f" {'Latency':>8} {'Calls':>5} {'Tok/step':>8}"
            f" {'Parse%':>7} {'Errors':>7}\n"
            f"{'─'*110}"
        )
        print(header)
        for method, rows in by_method.items():
            n   = len(rows)
            err = sum(1 for r in rows if r.error)
            print(
                f"  {method:<16} {rows[0].model:<14}"
                f" {avg(rows,'quality_score'):>7.3f}"
                f" {avg(rows,'efficiency_score'):>6.3f}"
                f" {avg(rows,'overall_score'):>8.3f}"
                f" {avg(rows,'latency_s'):>8.2f}s"
                f" {avg(rows,'llm_calls'):>5.1f}"
                f" {avg(rows,'tokens_per_step'):>8.0f}"
                f" {avg(rows,'parse_ok')*100:>7.1f}%"
                f" {err:>5}/{n}"
            )

    print(f"{'─'*110 if not run_sim else '─'*130}")


# ═════════════════════════════════════════════════════════════════════
# Main evaluation runner
# ═════════════════════════════════════════════════════════════════════

def run_sim_evaluation(
    dataset_path:  str,
    methods:       list[str],
    host:          str,
    model:         str,
    provider:      str,
    api_key:       str,
    sim_host:      str,
    sim_port:      int,
    out_path:      str,
    run_sim:       bool,
    dry_run_llm:   bool,
    max_replan:    int,
    max_samples:   int | None,
    verbose:       bool,
    method_kwargs: dict[str, dict],
) -> str:

    # ── Load dataset ──────────────────────────────────────────────────
    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)
    samples = data["samples"] if "samples" in data else data
    if max_samples:
        samples = samples[:max_samples]
    print(f"\n📂  Loaded {len(samples)} samples from {dataset_path}")

    # ── LLM connection check ──────────────────────────────────────────
    if not dry_run_llm:
        ok, msg = check_connection(provider, host, model, api_key)
        if ok:
            print(f"  ✅  LLM OK — {msg}")
        else:
            print(f"\n{'═'*60}\n  ❌  LLM CONNECTION FAILED\n{'─'*60}")
            for line in msg.splitlines():
                print(f"  {line}")
            print(f"{'═'*60}\n  Tip: use --dry-run-llm to skip LLM calls\n")
            sys.exit(1)

    # ── Simulator connection check ─────────────────────────────────────
    client = None
    if run_sim:
        ok, msg = check_simulator(sim_host, sim_port)
        if ok:
            print(f"  ✅  Simulator OK — {msg}")
            client = ThorClient(host=sim_host, port=sim_port)
        else:
            print(f"\n{'═'*60}\n  ❌  SIMULATOR NOT RUNNING\n{'─'*60}")
            for line in msg.splitlines():
                print(f"  {line}")
            print(f"{'═'*60}")
            print("  Continuing with --no-sim (offline metrics only)...")
            run_sim = False

    # ── Build planners ────────────────────────────────────────────────
    planners = []
    if dry_run_llm:
        stub      = _DryRunPlanner()
        stub.name = "DryRun"
        planners  = [stub]
        print("🔧  Dry-run LLM mode")
    else:
        for name in methods:
            kwargs = {**method_kwargs.get(name, {}), "provider": provider, "api_key": api_key}
            try:
                p = pyplanner.get(name, host=host, model=model, **kwargs)
                planners.append(p)
                print(f"  ✅  {name:15}  ({provider}/{model})")
            except Exception as e:
                print(f"  ❌  {name}: {e}")

    if not planners:
        print("No planners initialised — aborting.")
        return ""

    mode_label = "plan+execute" if run_sim else "plan only"
    print(f"\n  Mode: {mode_label}  |  max_replan={max_replan}")

    # ── Run evaluation ────────────────────────────────────────────────
    all_results: list[SimSampleResult] = []
    total = len(planners) * len(samples)
    done  = 0

    for planner in planners:
        print(f"\n▶  [{planner.name}]  {len(samples)} samples  ({mode_label})")
        consecutive_errors = 0

        for sample in samples:
            result = evaluate_sample_sim(
                client     = client,
                planner    = planner,
                sample     = sample,
                max_replan = max_replan,
                run_sim    = run_sim,
                verbose    = verbose,
            )
            all_results.append(result)
            done += 1

            if result.error or result.sim_error:
                consecutive_errors += 1
                if consecutive_errors >= 3:
                    err = result.error or result.sim_error
                    print(f"\n  ❌  3 consecutive errors — aborting [{planner.name}]")
                    print(f"     Last: {err[:80]}")
                    break
            else:
                consecutive_errors = 0

            if not verbose:
                pct = done / total * 100
                bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
                print(f"\r  [{bar}] {pct:5.1f}%  {done}/{total}", end="", flush=True)

        if not verbose:
            print()

    # ── Write CSV ─────────────────────────────────────────────────────
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SIM_CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in all_results:
            writer.writerow(r.to_row())

    _print_summary(all_results, run_sim)
    print(f"\n✅  Results saved → {out_path}")
    print(f"    Rows: {len(all_results)}  ·  Columns: {len(SIM_CSV_COLUMNS)}\n")
    return out_path


# ═════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════

def main():
    all_methods = list(REGISTRY.keys())

    parser = argparse.ArgumentParser(
        description="Evaluate pyplanner methods by executing plans in AI2-THOR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset",   default="eval_dataset.json")
    parser.add_argument("--methods",   nargs="+", default=all_methods)
    parser.add_argument("--host",      default=DEFAULT_HOST)
    parser.add_argument("--model",     default=DEFAULT_MODEL)
    parser.add_argument("--provider",  default=DEFAULT_BACKEND,
                        choices=["ollama","openai","anthropic"])
    parser.add_argument("--api-key",   default="")
    parser.add_argument("--sim-host",  default="localhost",
                        help="ThorServer host (default: localhost)")
    parser.add_argument("--sim-port",  type=int, default=5555,
                        help="ThorServer ZMQ port (default: 5555)")
    parser.add_argument("--out",       default="sim_results.csv")
    parser.add_argument("--no-sim",    action="store_true",
                        help="Skip simulator execution (offline metrics only)")
    parser.add_argument("--dry-run-llm", action="store_true",
                        help="Stub LLM calls (still executes in simulator if available)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit number of samples (useful for quick tests)")
    parser.add_argument("--max-replan",  type=int, default=3)
    parser.add_argument("--verbose",   action="store_true")
    parser.add_argument("--react-max-steps",   type=int, default=15)
    parser.add_argument("--refine-iterations", type=int, default=2)
    args = parser.parse_args()

    methods = all_methods if "all" in args.methods else args.methods
    method_kwargs = {
        "ReAct":       {"max_steps":      args.react_max_steps},
        "Self-Refine": {"max_iterations": args.refine_iterations},
    }

    # Auto-generate dataset if missing
    if not os.path.exists(args.dataset):
        print(f"⚠  Dataset not found: {args.dataset} — generating...")
        samples  = build_dataset()
        warnings = validate_dataset(samples)
        for w in warnings:
            print(f"   ⚠  {w}")
        with open(args.dataset, "w", encoding="utf-8") as f:
            json.dump({"version": "1.0", "samples": samples}, f, indent=2, ensure_ascii=False)
        print(f"   ✅  Generated {len(samples)} samples → {args.dataset}")

    print(f"\n{'═'*65}")
    print(f"  PyPlanner Simulation Evaluation")
    print(f"{'═'*65}")
    print(f"  Methods    : {methods}")
    print(f"  Provider   : {args.provider}  |  Model: {args.model}")
    print(f"  LLM host   : {args.host}")
    print(f"  Sim host   : {args.sim_host}:{args.sim_port}")
    print(f"  Run sim    : {'no (offline only)' if args.no_sim else 'yes'}")
    print(f"  Output     : {args.out}")
    if args.max_samples:
        print(f"  Max samples: {args.max_samples}")
    print(f"{'═'*65}")

    t_start = time.perf_counter()
    out = run_sim_evaluation(
        dataset_path  = args.dataset,
        methods       = methods,
        host          = args.host,
        model         = args.model,
        provider      = args.provider,
        api_key       = args.api_key,
        sim_host      = args.sim_host,
        sim_port      = args.sim_port,
        out_path      = args.out,
        run_sim       = not args.no_sim,
        dry_run_llm   = args.dry_run_llm,
        max_replan    = args.max_replan,
        max_samples   = args.max_samples,
        verbose       = args.verbose,
        method_kwargs = method_kwargs,
    )
    if out:
        print(f"  Total time: {time.perf_counter() - t_start:.1f}s\n")


if __name__ == "__main__":
    main()