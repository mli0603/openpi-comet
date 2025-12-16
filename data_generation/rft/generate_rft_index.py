#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple


def load_task_mapping(tasks_jsonl_path: Path) -> Dict[str, int]:
    mapping = {}
    with tasks_jsonl_path.open("r") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                mapping[data["task_name"]] = int(data["task_index"])
    return mapping


def get_task_name(exp_dir: Path) -> Optional[str]:
    """Parse task_name from cfg*.out file located directly under exp_dir."""
    for cfg_file in sorted(exp_dir.glob("cfg*.out")):
        try:
            # Read line by line to avoid loading huge files if they exist
            with cfg_file.open("r") as f:
                for line in f:
                    if line.strip().startswith("task_name="):
                        return line.split("=", 1)[1].strip()
        except Exception:
            continue
    return None


def is_valid_run(run_dir: Path) -> bool:
    """Check if run has a non-empty state_action.npz."""
    npz_path = run_dir / "state_action.npz"
    return npz_path.is_file() and npz_path.stat().st_size > 0


def find_runs(root_dir: Path) -> Iterator[Tuple[str, str, str, Path]]:
    """
    Yields (date_dir_name, exp_name, run_name, run_dir_path).
    Supports two structures:
      1. root/date/exp/rollouts/run
      2. root/exp/rollouts/run (where root is treated as the date dir)
    """
    # Try Pattern 1: root/*/*/rollouts/*
    pattern_1_candidates = sorted(root_dir.glob("*/*/rollouts/*"))
    if pattern_1_candidates:
        for run_path in pattern_1_candidates:
            if run_path.is_dir():
                # run_path = root/date/exp/rollouts/run
                exp_dir = run_path.parent.parent
                date_dir = exp_dir.parent
                yield date_dir.name, exp_dir.name, run_path.name, run_path
        return

    # Fallback to Pattern 2: root/*/rollouts/*
    pattern_2_candidates = sorted(root_dir.glob("*/rollouts/*"))
    for run_path in pattern_2_candidates:
        if run_path.is_dir():
            # run_path = root/exp/rollouts/run
            exp_dir = run_path.parent.parent
            # In this case, root_dir is treated as the date directory
            yield root_dir.name, exp_dir.name, run_path.name, run_path


def scan_existing_indices(
    output_dir: Path, exclude_file: Optional[Path], prefix: str = "rollouts"
) -> Dict[int, int]:
    """
    Scans existing JSONL files to find the maximum episode index for each task base.
    Returns {base_id: max_current_index}.
    """
    base_max_map: Dict[int, int] = {}
    if not output_dir.is_dir():
        return base_max_map

    for jsonl_file in output_dir.glob("*.jsonl"):
        if exclude_file and jsonl_file.resolve() == exclude_file.resolve():
            continue
        if not jsonl_file.name.startswith(prefix):
            continue

        try:
            with jsonl_file.open("r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.rsplit(" ", 1)
                    if len(parts) != 2:
                        continue
                    
                    episode_id_str = parts[1]
                    if not episode_id_str.isdigit() or len(episode_id_str) < 4:
                        continue
                    
                    eid = int(episode_id_str)
                    base = eid // 10000
                    idx = eid % 10000
                    base_max_map[base] = max(base_max_map.get(base, -1), idx)
        except Exception:
            continue
            
    return base_max_map


def main():
    parser = argparse.ArgumentParser(description="Dump rollout entries to RFT dataset index")
    parser.add_argument("root_dir", type=Path, help="Rollouts root directory to scan")
    parser.add_argument("output_jsonl", type=Path, help="Output RFT dataset JSONL path")
    parser.add_argument(
        "--tasks_jsonl",
        type=Path,
        default=Path("2025-challenge-demos/meta/tasks.jsonl"),
        help="Path to BEHAVIOR tasks.jsonl file",
    )
    args = parser.parse_args()

    if not args.tasks_jsonl.exists():
        raise FileNotFoundError(f"Tasks JSONL file not found: {args.tasks_jsonl}")

    task_map = load_task_mapping(args.tasks_jsonl)

    # Collect all valid runs first
    valid_runs = []
    for date_name, exp_name, run_name, run_path in find_runs(args.root_dir):
        if is_valid_run(run_path):
            valid_runs.append((date_name, exp_name, run_name, run_path))

    if not valid_runs:
        print(f"No valid rollout runs found under {args.root_dir}")
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        args.output_jsonl.touch()
        return

    # Cache task names for experiments
    exp_to_task = {}
    for _, _, _, run_path in valid_runs:
        # rollouts_dir is run_path.parent, exp_dir is run_path.parent.parent
        exp_dir = run_path.parent.parent
        if exp_dir not in exp_to_task:
            task_name = get_task_name(exp_dir)
            if task_name:
                exp_to_task[exp_dir] = task_name

    # Determine starting indices based on other files
    output_dir = args.output_jsonl.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    base_indices = scan_existing_indices(output_dir, exclude_file=args.output_jsonl)

    # Generate new entries
    new_lines = []
    # Deduplicate by relative path key to match original logic logic
    processed_rels = set()
    
    for date_name, exp_name, run_name, run_path in valid_runs:
        rel_key = f"{date_name}/{exp_name}/{run_name}"
        if rel_key in processed_rels:
            continue
        processed_rels.add(rel_key)

        exp_dir = run_path.parent.parent
        task_name = exp_to_task.get(exp_dir)
        
        if not task_name or task_name not in task_map:
            # Skip if task name not found or not in mapping
            continue

        base = task_map[task_name]
        next_idx = base_indices.get(base, -1) + 1

        if next_idx > 9999:
            print(f"WARNING: Index overflow for {task_name} (base={base}). Skipping {rel_key}")
            continue

        base_indices[base] = next_idx
        episode_id = f"{base:04d}{next_idx:04d}"
        new_lines.append(f"{rel_key} {episode_id}\n")

    with args.output_jsonl.open("w") as f:
        f.writelines(new_lines)

    print(f"Generated {len(new_lines)} entries in {args.output_jsonl}")


if __name__ == "__main__":
    main()
