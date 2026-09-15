"""Compare steady-state OPD timing records without averaging per-batch rates."""

import argparse
import json
from pathlib import Path


def load_records(path):
    path = Path(path)
    if path.is_dir():
        path /= "opd_metrics.jsonl"
    with path.open() as source:
        records = []
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            if "opd/resume_from" in record:
                records = [r for r in records if r["opd/batch_id"] < record["opd/resume_from"]]
            records.append(record)
        return records


def _steady(records, objective):
    selected = [r for r in records if r.get("opd/phase") == "steady"]
    if not selected:
        raise ValueError(f"No steady-state records for {objective}")
    if any(r["opd/objective"] != objective for r in selected):
        raise ValueError(f"Expected only {objective} records")
    ids = [r["opd/batch_id"] for r in selected]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate batch IDs; select one run rather than combining resumed or repeated batches")
    return selected


def _workload(records):
    return [
        (
            r["opd/batch_id"],
            r.get("opd/workload_hash"),
            r["opd/effective_tokens"],
            r["opd/computed_tokens"],
            r.get("opd/conditions"),
        )
        for r in records
    ]


def summarize(records):
    tokens = sum(r["opd/effective_tokens"] for r in records)
    if tokens <= 0:
        raise ValueError("The comparison needs positive effective token counts")
    summary = {
        "batches": len(records),
        "effective_tokens": tokens,
        "computed_tokens": sum(r["opd/computed_tokens"] for r in records),
    }
    for stage in ("learner", "e2e", "learner_compute", "head_loss"):
        key = f"perf/opd_{stage}_s"
        measured = [r for r in records if key in r]
        if measured and (len(measured) == len(records) or stage in {"learner_compute", "head_loss"}):
            seconds = sum(r[key] for r in measured)
            measured_tokens = sum(r["opd/effective_tokens"] for r in measured)
            if not measured_tokens:
                continue
            if seconds <= 0:
                raise ValueError(f"{key} must have positive total elapsed time")
            summary[stage] = {
                "measured_batches": len(measured),
                "effective_tokens": measured_tokens,
                "seconds": seconds,
                "ms_per_1k_tokens": seconds * 1_000_000 / measured_tokens,
                "effective_tokens_per_second": measured_tokens / seconds,
            }
    memory_key = "perf/opd_peak_allocated_gib"
    if all(memory_key in r for r in records):
        summary["peak_allocated_gib"] = max(r[memory_key] for r in records)
    return summary


def compare_records(sampled, full, *, allow_different_workloads=False):
    sampled = _steady(sampled, "sampled")
    full = _steady(full, "full_vocab_reverse_kl")
    benchmark_ids = {r.get("opd/benchmark_id") for r in sampled + full}
    matched = (
        all(r.get("opd/workload_hash") for r in sampled + full)
        and all(r.get("opd/conditions", {}).get("teacher_identity") for r in sampled + full)
        and _workload(sampled) == _workload(full)
        and len(benchmark_ids) == 1
    )
    if not matched and not allow_different_workloads:
        raise ValueError(
            "Workloads differ or have no workload hash/teacher identity. Replay the same trajectories/settings; use --allow-different-workloads only for an uncontrolled online comparison."
        )
    baseline, candidate = summarize(sampled), summarize(full)
    differences = {}
    for stage in ("learner", "e2e", "learner_compute", "head_loss"):
        if stage in baseline and stage in candidate:
            key = f"perf/opd_{stage}_s"
            if not allow_different_workloads and [r["opd/batch_id"] for r in sampled if key in r] != [
                r["opd/batch_id"] for r in full if key in r
            ]:
                raise ValueError(f"Profiled batch IDs differ for {stage}; use the same GPU timing interval")
            ratio = candidate[stage]["ms_per_1k_tokens"] / baseline[stage]["ms_per_1k_tokens"]
            differences[stage] = {
                "time_multiplier": ratio,
                "overhead_percent": (ratio - 1) * 100,
                "throughput_drop_percent": (1 - 1 / ratio) * 100,
            }
    if not differences:
        raise ValueError("No common learner or end-to-end timing metric")
    if "peak_allocated_gib" in baseline and "peak_allocated_gib" in candidate:
        differences["peak_memory_delta_gib"] = candidate["peak_allocated_gib"] - baseline["peak_allocated_gib"]
    return {
        "matched_workload": bool(matched),
        "sampled": baseline,
        "full_vocab_reverse_kl": candidate,
        "comparison": differences,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sampled", required=True, help="Sampled JSONL file or directory containing opd_metrics.jsonl"
    )
    parser.add_argument("--full", required=True, help="Full-vocabulary JSONL file or directory")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-different-workloads", action="store_true")
    args = parser.parse_args()
    report = compare_records(
        load_records(args.sampled), load_records(args.full), allow_different_workloads=args.allow_different_workloads
    )
    result = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result + "\n")
    print(result)


if __name__ == "__main__":
    main()
