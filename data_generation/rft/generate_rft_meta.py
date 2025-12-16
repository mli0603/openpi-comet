#!/usr/bin/env python3
import argparse
import json
import os
from typing import Tuple

import numpy as np
import pandas as pd
from omnigibson.learning.utils.eval_utils import PROPRIOCEPTION_INDICES
from tqdm import tqdm


def generate_task_json(jsonl: str, tasks_jsonl: str) -> int:
    # read index file
    with open(jsonl, "r") as f:
        lines = f.readlines()
    
    # convert episode index to task index
    all_tasks = set()
    for line in lines:
        _, episode_id = line.strip().split(" ")
        task_index = int(episode_id) // 10000
        all_tasks.add(task_index)
    
    # read tasks jsonl
    with open(tasks_jsonl, "r") as f:
        tasks = [json.loads(line) for line in f]
    
    # keep only tasks that are in the index
    task_json = []
    for task in tasks:
        if task["task_index"] in all_tasks:
            task_json.append(task)
    
    with open(f"{data_dir}/meta/tasks.jsonl", "w") as f:
        for task in task_json:
            json.dump(task, f)
            f.write("\n")
    
    num_tasks = len(task_json)
    print(f"Generated task JSON for {num_tasks} tasks.")
    return num_tasks


def generate_episode_json(data_dir: str, robot_type: str = "R1Pro") -> Tuple[int, int]:
    assert os.path.exists(f"{data_dir}/meta/tasks.jsonl"), "Task JSON does not exist!"
    assert os.path.exists(f"{data_dir}/meta/episodes"), "Episode metadata directory does not exist!"
    with open(f"{data_dir}/meta/tasks.jsonl", "r") as f:
        task_json = [json.loads(line) for line in f]
    num_frames = 0
    num_episodes = 0
    with open(f"{data_dir}/meta/episodes.jsonl", "w") as out_f:
        with open(f"{data_dir}/meta/episodes_stats.jsonl", "w") as out_stats_f:
            for task_info in tqdm(task_json):
                task_index = task_info["task_index"]
                task_name = task_info["task"]
                if not os.path.exists(f"{data_dir}/meta/episodes/task-{task_index:04d}"):
                    continue
                for episode_name in tqdm(sorted(os.listdir(f"{data_dir}/meta/episodes/task-{task_index:04d}"))):
                    with open(f"{data_dir}/meta/episodes/task-{task_index:04d}/{episode_name}", "r") as f:
                        episode_info = json.load(f)
                        episode_index = int(episode_name.split(".")[0].split("_")[-1])
                        episode_json = {
                            "episode_index": episode_index,
                            "tasks": [task_name],
                            "length": episode_info["num_samples"],
                        }
                        # load the corresponding parquet file
                        episode_df = pd.read_parquet(
                            f"{data_dir}/data/task-{task_index:04d}/episode_{episode_index:08d}.parquet"
                        )
                        episode_stats = {}
                        for key in ["action", "observation.state"]:
                            if key not in episode_stats:
                                episode_stats[key] = {}
                            values = np.stack(episode_df[key].values)
                            if len(values.shape) == 1:
                                values = values[:, np.newaxis]
                            episode_stats[key]["min"] = values.min(axis=0).tolist()
                            episode_stats[key]["max"] = values.max(axis=0).tolist()
                            episode_stats[key]["mean"] = values.mean(axis=0).tolist()
                            episode_stats[key]["std"] = values.std(axis=0).tolist()
                            episode_stats[key]["q01"] = np.quantile(values, 0.01, axis=0).tolist()
                            episode_stats[key]["q99"] = np.quantile(values, 0.99, axis=0).tolist()
                            episode_stats[key]["count"] = [values.shape[0]]
                            if key == "observation.state":
                                robot_pos = values[:, PROPRIOCEPTION_INDICES[robot_type]["robot_pos"]]
                                episode_json["distance_traveled"] = round(
                                    np.sum(np.linalg.norm(robot_pos[1:, :] - robot_pos[:-1, :], axis=-1)).item(), 4
                                )
                                left_eef_pos = values[:, PROPRIOCEPTION_INDICES[robot_type]["eef_left_pos"]]
                                right_eef_pos = values[:, PROPRIOCEPTION_INDICES[robot_type]["eef_right_pos"]]
                                episode_json["left_eef_displacement"] = round(
                                    np.sum(np.linalg.norm(left_eef_pos[1:, :] - left_eef_pos[:-1, :], axis=-1)).item(),
                                    4,
                                )
                                episode_json["right_eef_displacement"] = round(
                                    np.sum(
                                        np.linalg.norm(right_eef_pos[1:, :] - right_eef_pos[:-1, :], axis=-1)
                                    ).item(),
                                    4,
                                )
                        episode_stats_json = {
                            "episode_index": episode_index,
                            "stats": episode_stats,
                        }
                        num_episodes += 1
                        num_frames += episode_info["num_samples"]
                    json.dump(episode_json, out_f)
                    out_f.write("\n")
                    json.dump(episode_stats_json, out_stats_f)
                    out_stats_f.write("\n")
    print(f"Generated episode JSON for {num_episodes} episodes and {num_frames} frames.")
    return num_episodes, num_frames


def generate_info_json(
    data_dir: str,
    fps: int = 30,
    total_episodes: int = 50,
    total_tasks: int = 50,
    total_frames: int = 50,
):
    info = {
        "codebase_version": "v2.1",
        "robot_type": "R1Pro",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
        "total_videos": total_episodes * 9,
        "chunks_size": 10000,
        "fps": fps,
        "splits": {
            "train": "0:" + str(total_episodes),
        },
        "data_path": "data/task-{episode_chunk:04d}/episode_{episode_index:08d}.parquet",
        "video_path": "videos/task-{episode_chunk:04d}/{video_key}/episode_{episode_index:08d}.mp4",
        "metainfo_path": "meta/episodes/task-{episode_chunk:04d}/episode_{episode_index:08d}.json",
        "annotation_path": "annotations/task-{episode_chunk:04d}/episode_{episode_index:08d}.json",
        "features": {
            "observation.images.rgb.left_wrist": {
                "dtype": "video",
                "shape": [480, 480, 3],
                "names": ["height", "width", "rgb"],
                "info": {
                    "video.fps": 30.0,
                    "video.height": 480,
                    "video.width": 480,
                    "video.channels": 3,
                    "video.codec": "libx265",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            },
            "observation.images.rgb.right_wrist": {
                "dtype": "video",
                "shape": [480, 480, 3],
                "names": ["height", "width", "rgb"],
                "info": {
                    "video.fps": 30.0,
                    "video.height": 480,
                    "video.width": 480,
                    "video.channels": 3,
                    "video.codec": "libx265",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            },
            "observation.images.rgb.head": {
                "dtype": "video",
                "shape": [720, 720, 3],
                "names": ["height", "width", "rgb"],
                "info": {
                    "video.fps": 30.0,
                    "video.height": 720,
                    "video.width": 720,
                    "video.channels": 3,
                    "video.codec": "libx265",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            },
            "observation.images.depth.left_wrist": {
                "dtype": "video",
                "shape": [480, 480, 3],
                "names": ["height", "width", "depth"],
                "info": {
                    "video.fps": 30.0,
                    "video.height": 480,
                    "video.width": 480,
                    "video.channels": 3,
                    "video.codec": "libx265",
                    "video.pix_fmt": "yuv420p16le",
                    "video.is_depth_map": True,
                    "has_audio": False,
                },
            },
            "observation.images.depth.right_wrist": {
                "dtype": "video",
                "shape": [480, 480, 3],
                "names": ["height", "width", "depth"],
                "info": {
                    "video.fps": 30.0,
                    "video.height": 480,
                    "video.width": 480,
                    "video.channels": 3,
                    "video.codec": "libx265",
                    "video.pix_fmt": "yuv420p16le",
                    "video.is_depth_map": True,
                    "has_audio": False,
                },
            },
            "observation.images.depth.head": {
                "dtype": "video",
                "shape": [720, 720, 3],
                "names": ["height", "width", "depth"],
                "info": {
                    "video.fps": 30.0,
                    "video.height": 720,
                    "video.width": 720,
                    "video.channels": 3,
                    "video.codec": "libx265",
                    "video.pix_fmt": "yuv420p16le",
                    "video.is_depth_map": True,
                    "has_audio": False,
                },
            },
            "observation.images.seg_instance_id.left_wrist": {
                "dtype": "video",
                "shape": [480, 480, 3],
                "names": ["height", "width", "rgb"],
                "info": {
                    "video.fps": 30.0,
                    "video.height": 480,
                    "video.width": 480,
                    "video.channels": 3,
                    "video.codec": "libx265",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            },
            "observation.images.seg_instance_id.right_wrist": {
                "dtype": "video",
                "shape": [480, 480, 3],
                "names": ["height", "width", "rgb"],
                "info": {
                    "video.fps": 30.0,
                    "video.height": 480,
                    "video.width": 480,
                    "video.channels": 3,
                    "video.codec": "libx265",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            },
            "observation.images.seg_instance_id.head": {
                "dtype": "video",
                "shape": [720, 720, 3],
                "names": ["height", "width", "rgb"],
                "info": {
                    "video.fps": 30.0,
                    "video.height": 720,
                    "video.width": 720,
                    "video.channels": 3,
                    "video.codec": "libx265",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            },
            "action": {"dtype": "float32", "shape": [23], "names": None},
            "timestamp": {"dtype": "float64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "observation.cam_rel_poses": {"dtype": "float32", "shape": [21], "names": None},
            "observation.state": {"dtype": "float32", "shape": [256], "names": None},
            "observation.task_info": {"dtype": "float32", "shape": [None], "names": None},
        },
    }

    with open(f"{data_dir}/meta/info.json", "w") as f:
        json.dump(info, f, indent=4)

    print(f"Generated info JSON for {len(info)} entries.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", type=str, help="Data directory")
    parser.add_argument("--jsonl", type=str, help="jsonl file to the index generated by convert_rft_data.py")
    parser.add_argument("--tasks-jsonl", type=str, help="Task JSONL path")
    args = parser.parse_args()

    # expand root
    data_dir = os.path.expanduser(args.data_dir)
    print("Generating task JSON...")
    num_tasks = generate_task_json(args.jsonl, args.tasks_jsonl)
    print("Generating episode JSON...")
    num_episodes, num_frames = generate_episode_json(data_dir)
    print(num_tasks, num_episodes, num_frames)
    print("Generating info JSON...")
    generate_info_json(data_dir, fps=30, total_episodes=num_episodes, total_tasks=num_tasks, total_frames=num_frames)