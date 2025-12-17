#!/usr/bin/env python3
import argparse
import functools
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import tempfile
from typing import Dict
from typing import Dict as _Dict
from typing import List, Tuple, cast

import numpy as np
import pandas as pd

# Constants
TRIM_FRAMES = 20
DEFAULT_FPS = 30

def load_task_mapping(tasks_jsonl_path: str) -> Dict[str, int]:
    task_name_to_index: Dict[str, int] = {}
    with open(tasks_jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            task_name_to_index[data["task_name"]] = int(data["task_index"])
    return task_name_to_index


def read_task_name_from_exp(exp_dir: str) -> str | None:
    # cfg*.out under exp_dir; find line 'task_name=*'
    if not os.path.isdir(exp_dir):
        return None
    for fname in sorted(os.listdir(exp_dir)):
        if fname.startswith("cfg") and fname.endswith(".out"):
            path = os.path.join(exp_dir, fname)
            try:
                with open(path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("task_name="):
                            return line.split("=", 1)[1].strip()
            except Exception:
                continue
    return None


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_jsonl_line(line: str) -> Tuple[str, int]:
    # "<date>/<exp>/<run> <episode_id>"
    rel, eid_str = line.strip().rsplit(" ", 1)
    return rel, int(eid_str)


def load_npz_state_action(npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.load(npz_path, allow_pickle=True)
    # Updated format:
    # arr_0 is a 0-d object array holding a dict with 'state' and 'action'
    obj = arr["arr_0"].item()
    state = np.asarray(obj["state"])
    action = np.asarray(obj["action"])
    # Normalize shapes to 2D [T, D]
    if state.ndim == 3 and state.shape[1] == 1:
        state = state[:, 0, :]
    if action.ndim == 3 and action.shape[1] == 1:
        action = action[:, 0, :]
    if action.ndim == 3 and action.shape[-1] == 1:
        action = action[:, :, 0]
    if state.ndim != 2:
        raise ValueError(f"Expected state to be 2D, got shape {state.shape}")
    if action.ndim != 2:
        raise ValueError(f"Expected action to be 2D, got shape {action.shape}")
    if state.shape[0] != action.shape[0]:
        raise ValueError(f"State/action length mismatch at {npz_path}: {state.shape} vs {action.shape}")
    return state, action


def ensure_dir_fs(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_json_fs(path: str, obj: dict) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)


def copy_file_to_fs(src_local_path: str, dst_path: str) -> None:
    shutil.copyfile(src_local_path, dst_path)


def write_parquet(data_dir: str, episode_id: int, task_index: int, state: np.ndarray, action: np.ndarray) -> None:
    T = state.shape[0]
    # Ensure columns are list-of-lists (not numpy arrays) for parquet compatibility
    state_rows = [np.asarray(row, dtype=np.float32).tolist() for row in state]
    action_rows = [np.asarray(row, dtype=np.float32).tolist() for row in action]
    data = {
        "index": np.arange(T, dtype=np.int64),
        "episode_index": np.zeros(T, dtype=np.int64) + episode_id,
        "task_index": np.zeros(T, dtype=np.int64) + task_index,
        "timestamp": np.arange(T, dtype=np.float64) / 30.0,
        "observation.state": state_rows,
        "action": action_rows,
        "observation.task_info": [[] for _ in range(T)],  # dummy per-frame list
    }
    df = pd.DataFrame(data)
    out_path = os.path.join(data_dir, f"episode_{episode_id:08d}.parquet")
    df.to_parquet(out_path, index=False)


def copy_meta(template_root: str, out_root: str, task_index: int, episode_id: int) -> None:
    task_id_str = f"task-{task_index:04d}"
    # Find the first available meta json in the task directory
    src_dir = os.path.join(template_root, "meta", "episodes", task_id_str)
    if not os.path.isdir(src_dir):
        # Graceful fallback or detailed error? raise for now as in original
        raise FileNotFoundError(f"Meta source dir not found: {src_dir}")
    candidates = [f for f in sorted(os.listdir(src_dir)) if f.endswith(".json")]
    if not candidates:
        raise FileNotFoundError(f"No meta json found under: {src_dir}")
    src = os.path.join(src_dir, candidates[0])
    dst_dir = os.path.join(out_root, "meta", "episodes", task_id_str)
    ensure_dir_fs(dst_dir)
    dst = os.path.join(dst_dir, f"episode_{episode_id:08d}.json")
    # Copy as-is
    copy_file_to_fs(src, dst)


def copy_and_update_annotation(template_root: str, out_root: str, task_index: int, episode_id: int, num_frames: int) -> None:
    task_id_str = f"task-{task_index:04d}"
    # Find the first available annotation json in the task directory
    src_dir = os.path.join(template_root, "annotations", task_id_str)
    if not os.path.isdir(src_dir):
        raise FileNotFoundError(f"Annotation source dir not found: {src_dir}")
    candidates = [f for f in sorted(os.listdir(src_dir)) if f.endswith(".json")]
    if not candidates:
        raise FileNotFoundError(f"No annotation json found under: {src_dir}")
    src = os.path.join(src_dir, candidates[0])
    dst_dir = os.path.join(out_root, "annotations", task_id_str)
    ensure_dir_fs(dst_dir)
    dst = os.path.join(dst_dir, f"episode_{episode_id:08d}.json")
    with open(src, "r") as f:
        ann = json.load(f)
    # Update meta_data durations
    ann.setdefault("meta_data", {})
    ann["meta_data"]["task_duration"] = int(num_frames)
    ann["meta_data"]["valid_duration"] = [0, int(num_frames)]
    # Keep only first skill, adjust its frame_duration to whole episode
    skills = ann.get("skill_annotation", [])
    if skills:
        first = skills[0]
        first["frame_duration"] = [0, int(num_frames)]
        ann["skill_annotation"] = [first]
    else:
        ann["skill_annotation"] = []
    write_json_fs(dst, ann)


def trim_and_copy_videos(
    run_dir: str,
    out_root: str,
    task_index: int,
    episode_id: int,
    num_frames_out: int,
    trim_frames: int = TRIM_FRAMES,
) -> None:
    task_id_str = f"task-{task_index:04d}"
    mapping = {
        "head.mp4": ("videos", task_id_str, "observation.images.rgb.head", f"episode_{episode_id:08d}.mp4"),
        "left_wrist.mp4": ("videos", task_id_str, "observation.images.rgb.left_wrist", f"episode_{episode_id:08d}.mp4"),
        "right_wrist.mp4": ("videos", task_id_str, "observation.images.rgb.right_wrist", f"episode_{episode_id:08d}.mp4"),
    }

    # trim frames from the start
    vf = f"select='gte(n,{TRIM_FRAMES})'"
    for src_name, path_parts in mapping.items():
        src = os.path.join(run_dir, src_name)
        if not os.path.exists(src):
            print(f"Warning: Source video not found: {src}", flush=True)
            continue
        dst_dir = os.path.join(out_root, *path_parts[:-1])
        ensure_dir_fs(dst_dir)
        dst = os.path.join(dst_dir, path_parts[-1])
        
        # Local destination: write to a tmp then move
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with tempfile.TemporaryDirectory() as td:
            tmp_out_ffmpeg = os.path.join(td, "trimmed_ffmpeg.mp4")
            ok_ffmpeg = _run_ffmpeg_trim(src, tmp_out_ffmpeg, vf)
            if ok_ffmpeg:
                shutil.copyfile(tmp_out_ffmpeg, dst)
            else:
                print(f"CRITICAL: ffmpeg trim also failed for {src}.", flush=True)
                raise RuntimeError(f"ffmpeg trim also failed for {src}.")

def _run_ffmpeg_trim(src: str, dst: str, vf_filter: str) -> bool:
    """
    Execute ffmpeg to trim first N frames and re-encode video.
    We drop audio for robustness (-an) to avoid failures on streams without audio.
    """
    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            src,
            "-vf",
            vf_filter,
            "-vsync",
            "vfr",
            "-an",
            dst,
        ]
        # Using check=True to raise on failure
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except Exception:
        return False


def exists_fs(path: str) -> bool:
    return os.path.exists(path)


def all_outputs_exist(out_root: str, task_index: int, episode_id: int) -> bool:
    base_task = f"task-{task_index:04d}"
    eid = f"{episode_id:08d}"
    files = [
        os.path.join(out_root, "data", base_task, f"episode_{eid}.parquet"),
        os.path.join(out_root, "meta", "episodes", base_task, f"episode_{eid}.json"),
        os.path.join(out_root, "annotations", base_task, f"episode_{eid}.json"),
        os.path.join(out_root, "videos", base_task, "observation.images.rgb.head", f"episode_{eid}.mp4"),
        os.path.join(out_root, "videos", base_task, "observation.images.rgb.left_wrist", f"episode_{eid}.mp4"),
        os.path.join(out_root, "videos", base_task, "observation.images.rgb.right_wrist", f"episode_{eid}.mp4"),
    ]
    return all(exists_fs(fp) for fp in files)


def process_job(
    job: _Dict[str, object],
    *,
    final_out_root: str,
    template_root: str,
) -> Tuple[str, int, str]:
    rel = cast(str, job["rel"])
    episode_id = int(cast(int, job["episode_id"]))
    exp_dir = cast(str, job["exp_dir"])
    run_dir = cast(str, job["run_dir"])
    npz_path = cast(str, job["npz_path"])

    # Derive task_index
    task_index = episode_id // 10000
    task_id_str = f"task-{task_index:04d}"

    # Prepare output dirs
    data_dir = os.path.join(final_out_root, "data", task_id_str)
    ensure_dir_fs(data_dir)

    # Load state/action
    try:
        state, action = load_npz_state_action(npz_path)
    except Exception as e:
        print(f"[error_load] {rel} -> {e}", flush=True)
        return ("error", episode_id, rel)
        
    # Trim off the first TRIM_FRAMES frames
    if state.shape[0] <= TRIM_FRAMES:
        print(f"[skip_too_short] episode {episode_id:08d} ({rel}) length={state.shape[0]} <= trim={TRIM_FRAMES}", flush=True)
        return ("skip_too_short", episode_id, rel)
    state = state[TRIM_FRAMES:, :]
    action = action[TRIM_FRAMES:, :]

    # Write parquet
    write_parquet(data_dir, episode_id, task_index, state, action)
    # Copy meta and annotations (update durations to number of frames)
    copy_meta(template_root, final_out_root, task_index, episode_id)
    # num_frames after trimming; timestamps in parquet start at 0, so annotations need duration = T_trimmed
    copy_and_update_annotation(template_root, final_out_root, task_index, episode_id, num_frames=state.shape[0])
    # Trim videos instead of raw copy
    trim_and_copy_videos(
        run_dir,
        final_out_root,
        task_index,
        episode_id,
        num_frames_out=state.shape[0],
        trim_frames=TRIM_FRAMES,
    )
    print(f"[converted] episode {episode_id:08d} ({rel}) -> {task_id_str}", flush=True)
    return ("converted", episode_id, rel)


def main():
    parser = argparse.ArgumentParser(description="Convert rollouts to rft dataset structure (Local Only)")
    parser.add_argument("--input-root", required=True, help="Rollouts root (local directory)")
    parser.add_argument("--output-root", required=True, help="Output dataset root (local directory)")
    parser.add_argument("--jsonl", required=True, help="JSONL produced by generate_rft_index.py")
    parser.add_argument("--tasks-jsonl", help="Path to tasks.jsonl")
    parser.add_argument("--template-root", default="2025-challenge-demos", help="Existing BEHAVIOR dataset to copy meta/annotation templates from")
    parser.add_argument("--num-workers", type=int, default=32, help="Parallel workers (default: half of CPU cores)")
    args = parser.parse_args()

    tasks_jsonl_path = args.tasks_jsonl
    template_root = args.template_root
    if not os.path.exists(tasks_jsonl_path):
        raise FileNotFoundError(f"Tasks JSONL not found: {tasks_jsonl_path}")
    if not os.path.isdir(template_root):
        raise FileNotFoundError(f"Template dataset root not found: {template_root}")

    task_name_to_index = load_task_mapping(tasks_jsonl_path)

    # Contain outputs by JSONL filename (without extension)
    jsonl_base = os.path.splitext(os.path.basename(args.jsonl))[0]
    final_out_root = os.path.join(args.output_root, jsonl_base)
    ensure_dir_fs(final_out_root)

    # Build jobs list
    jobs: List[_Dict[str, object]] = []
    with open(args.jsonl, "r") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            rel, episode_id = parse_jsonl_line(raw)
            date_part, exp_name, run_name = rel.split("/", 2)
            exp_dir = os.path.join(args.input_root, date_part, exp_name)
            run_dir = os.path.join(exp_dir, "rollouts", run_name)
            npz_path = os.path.join(run_dir, "state_action.npz")
            jobs.append(
                {
                    "rel": rel,
                    "episode_id": episode_id,
                    "exp_dir": exp_dir,
                    "run_dir": run_dir,
                    "npz_path": npz_path,
                }
            )

    # Worker wrapper with bound arguments for picklable multiprocessing
    worker = functools.partial(
        process_job,
        final_out_root=final_out_root,
        template_root=template_root,
    )

    # Execute in parallel
    if args.num_workers <= 1 or len(jobs) <= 1:
        results = [worker(j) for j in jobs]
    else:
        with mp.Pool(processes=args.num_workers) as pool:
            results = pool.map(worker, jobs)

    # Summarize
    converted = sum(1 for s, _, _ in results if s == "converted")

    total = len(results)
    print(f"Done. Total: {total}, converted: {converted}. Output at {final_out_root}")


if __name__ == "__main__":
    main()

