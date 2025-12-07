#!/usr/bin/env python3
"""
Local teleoperation script for OmniGibson BEHAVIOR-1K tasks.
Use keyboard to control the robot manually with full arm and base control.

Based on:
- OmniGibson's official robot_teleoperate_demo.py
- BEHAVIOR-1K evaluation framework (eval.sh, submit_openpi_eval.sh)

Controls:
- Robot arms: Arrow keys, I/K/J/L, etc. (TeleMoMa keyboard controls)
- Robot base: WASD keys
- Gripper: Open/Close with specific keys
- Quit: ESC or Ctrl+C
"""

import os
import sys
import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

try:
    import omnigibson as og
    from omnigibson.macros import gm
except ImportError:
    print("ERROR: OmniGibson not found. Please make sure you're in the correct conda environment.")
    sys.exit(1)


try:
    from telemoma.configs.base_config import teleop_config
    from omnigibson.utils.teleop_utils import TeleopSystem
except ImportError as e:
    print("ERROR: TeleMoMa not found. Install it with: pip install telemoma")
    print(f"Details: {e}")
    sys.exit(1)

try:
    from av.container import Container
    from av.stream import Stream
    from omnigibson.learning.utils.obs_utils import create_video_writer, write_video
    from omnigibson.learning.utils.eval_utils import ROBOT_CAMERA_NAMES, flatten_obs_dict
    import omnigibson.utils.transform_utils as T
    from flask import Flask, Response, render_template_string
    from werkzeug.serving import WSGIRequestHandler
    import threading
    import queue
    import logging
except ImportError as e:
    print(f"WARNING: Some video utilities not found. Video recording may not work: {e}")


class VideoStreamer:
    """
    Video streaming server using Flask and MJPEG.
    Allows real-time visualization through a web browser.
    """

    def __init__(self, host="0.0.0.0", port=5000):
        self.host = host
        self.port = port
        self.frame_queue = queue.Queue(maxsize=10)
        self.app = Flask(__name__)
        self.server_thread = None
        self.latest_frame = None
        self._setup_routes()

    def _setup_routes(self):
        """Setup Flask routes for video streaming."""

        @self.app.route("/")
        def index():
            """Video streaming home page."""
            html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>OmniGibson Teleoperation Stream</title>
                <style>
                    body {
                        margin: 0;
                        padding: 20px;
                        background-color: #1e1e1e;
                        font-family: Arial, sans-serif;
                        color: #ffffff;
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                    }
                    h1 { margin-bottom: 20px; }
                    img {
                        max-width: 100%;
                        border: 2px solid #444;
                        border-radius: 8px;
                    }
                    .info {
                        margin-top: 20px;
                        padding: 10px;
                        background-color: #2d2d2d;
                        border-radius: 5px;
                    }
                </style>
            </head>
            <body>
                <h1>OmniGibson Teleoperation Stream</h1>
                <img src="/video_feed" alt="Video Stream">
                <div class="info">
                    <p>Real-time video from robot cameras</p>
                    <p>Control the robot in the terminal</p>
                </div>
            </body>
            </html>
            """
            return render_template_string(html)

        @self.app.route("/video_feed")
        def video_feed():
            """Video streaming route. Returns MJPEG stream."""
            return Response(self._generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

    def _generate_frames(self):
        """Generator function for MJPEG streaming."""
        while True:
            try:
                # Try to get a fresh frame
                try:
                    frame = self.frame_queue.get(timeout=1.0)
                    self.latest_frame = frame
                except queue.Empty:
                    if self.latest_frame is None:
                        import time

                        time.sleep(0.05)
                        continue
                    frame = self.latest_frame

                # Encode frame as JPEG
                ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not ret:
                    continue

                frame_bytes = buffer.tobytes()

                # Yield frame in MJPEG format
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(frame_bytes)).encode() + b"\r\n\r\n" + frame_bytes + b"\r\n"
                )
            except Exception as e:
                logging.error(f"Error in frame generation: {e}")
                continue

    def push_frame(self, frame):
        """Push a new frame to the streaming queue."""
        if frame is None:
            return

        # Convert to uint8 BGR if needed
        if frame.dtype != np.uint8:
            if np.issubdtype(frame.dtype, np.floating):
                if frame.max() <= 1.0:
                    frame = (frame * 255.0).round()
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        # Ensure contiguous
        frame = np.ascontiguousarray(frame)

        # Clear old frames if queue is full
        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass

        try:
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            pass

    def start(self):
        """Start the Flask server in a separate thread."""
        # Disable Flask's logging
        log = logging.getLogger("werkzeug")
        log.setLevel(logging.ERROR)

        def run_server():
            WSGIRequestHandler.protocol_version = "HTTP/1.1"
            self.app.run(host=self.host, port=self.port, debug=False, threaded=True, use_reloader=False)

        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()

        print("")
        print("=" * 60)
        print(f"Video streaming server started!")
        print(f"   Open browser: http://localhost:{self.port}")
        print("=" * 60)
        print("")

    def stop(self):
        """Stop the streaming server."""
        pass  # Daemon thread will terminate with main program


class TeleopController:
    """Keyboard teleoperation controller for OmniGibson BEHAVIOR-1K tasks."""

    def __init__(
        self,
        task_name: str,
        headless: bool = False,
        behavior_base: Optional[str] = None,
        show_marker: bool = True,
        max_steps: Optional[int] = None,
        write_video: bool = True,
        log_path: Optional[str] = None,
        stream_port: int = 5000,
    ):
        """
        Initialize the teleoperation controller.

        Args:
            task_name: Name of the BEHAVIOR task (e.g., 'turning_on_radio', 'picking_up_trash')
            headless: Whether to run in headless mode (no GUI)
            behavior_base: Base directory for BEHAVIOR, defaults to environment or parent dir
            show_marker: Whether to show green control markers for end effectors
            max_steps: Maximum steps per episode, None for no limit
            write_video: Whether to record video of the teleoperation session
            log_path: Path to save videos and logs, defaults to behavior_base/outputs/teleop_videos
        """
        self.task_name = task_name
        self.headless = headless
        self.show_marker = show_marker
        self.max_steps = max_steps
        self.write_video = write_video
        self.stream_port = stream_port
        self.enable_streaming = stream_port is not None

        # Set behavior base directory
        if behavior_base is None:
            behavior_base = os.environ.get("BEHAVIOR_BASE", "/behavior")
        self.behavior_base = behavior_base

        # Set log path for videos
        if log_path is None:
            log_path = os.path.join(behavior_base, "outputs/teleop_videos")
        self.log_path = Path(log_path)

        # Set omnigibson to headless mode if requested
        if self.headless:
            os.environ["OMNIGIBSON_HEADLESS"] = "1"
            gm.HEADLESS = True

        # Enable object states for BEHAVIOR tasks
        gm.ENABLE_OBJECT_STATES = True
        gm.USE_GPU_DYNAMICS = False
        gm.ENABLE_TRANSITION_RULES = True
        gm.ENABLE_FLATCACHE = True

        print("=" * 80)
        print(f"Initializing BEHAVIOR-1K Teleoperation")
        print("=" * 80)
        print(f"Task: {task_name}")
        print(f"Headless mode: {headless}")
        print(f"BEHAVIOR base: {behavior_base}")
        print(f"Show control markers: {show_marker}")
        print(f"Write video: {write_video}")
        if write_video:
            print(f"Video output path: {log_path}")
        print("=" * 80)

        self.env = None
        self.robot = None
        self.teleop_sys = None
        self._video_writer = None
        self.obs = None

    def setup_environment(self):
        """Set up the OmniGibson environment with the specified BEHAVIOR task."""
        print("\n=== Setting up OmniGibson environment ===")

        # Navigate to BEHAVIOR-1K directory
        behavior_1k_path = os.path.join(self.behavior_base, "BEHAVIOR-1K")
        if not os.path.exists(behavior_1k_path):
            raise FileNotFoundError(f"BEHAVIOR-1K not found at {behavior_1k_path}")

        os.chdir(behavior_1k_path)
        sys.path.insert(0, os.path.join(behavior_1k_path, "OmniGibson"))

        print(f"Loading task: {self.task_name}")

        # Import required utilities
        from gello.robots.sim_robot.og_teleop_utils import (
            load_available_tasks,
            generate_basic_environment_config,
        )
        from omnigibson.learning.utils.eval_utils import PROPRIOCEPTION_INDICES

        # Load available tasks and get task config
        available_tasks = load_available_tasks()
        if self.task_name not in available_tasks:
            raise ValueError(f"Task '{self.task_name}' not found. Available tasks: {list(available_tasks.keys())}")

        # Load the seed instance by default
        task_cfg = available_tasks[self.task_name][0]

        # Generate environment configuration (similar to eval.py)
        cfg = generate_basic_environment_config(task_name=self.task_name, task_cfg=task_cfg)

        # Configure robot with teleoperation-friendly controllers
        # Use R1Pro robot (BEHAVIOR-1K standard)
        robot_cfg = {
            "type": "R1Pro",
            "obs_modalities": ["proprio", "rgb"],
            "proprio_obs": list(PROPRIOCEPTION_INDICES["R1Pro"].keys()),
            "action_normalize": False,  # Don't normalize for teleoperation
            "grasping_mode": "assisted",  # Simplified grasping
            "controller_config": {},
        }

        # Configure arm controllers with IK for easier control
        for arm in ["left", "right"]:
            robot_cfg["controller_config"][f"arm_{arm}"] = {
                "name": "InverseKinematicsController",
                "command_input_limits": None,  # No limits for teleoperation
            }
            robot_cfg["controller_config"][f"gripper_{arm}"] = {
                "name": "MultiFingerGripperController",
                "command_input_limits": (0.0, 1.0),
                "mode": "smooth",
            }

        cfg["robots"] = [robot_cfg]

        # Disable task observations (we don't need them for teleop)
        cfg["task"]["include_obs"] = False

        # Set max steps if provided
        if self.max_steps is not None:
            cfg["task"]["termination_config"]["max_steps"] = self.max_steps

        print("Creating OmniGibson environment...")
        print(f"  Scene: {cfg['scene']['type']}")
        print(f"  Robot: {cfg['robots'][0]['type']}")
        print(f"  Grasping mode: {cfg['robots'][0]['grasping_mode']}")

        # Create the environment
        self.env = og.Environment(configs=cfg)

        if self.env is None:
            raise RuntimeError("Failed to create OmniGibson environment")

        # Get the robot from the environment
        # Standard way: use env.robots list (same as BehaviorTask)
        if len(self.env.robots) == 0:
            raise RuntimeError("No robots found in environment")

        self.robot = self.env.robots[0]  # Get first robot

        print(f"\n✓ Robot loaded: {type(self.robot).__name__}")
        print(f"✓ Robot name: {self.robot.name}")
        print(f"✓ Robot action dimension: {self.robot.action_dim}")
        print("✓ Environment setup complete!\n")

    def setup_keyboard_controller(self):
        """Set up TeleMoMa keyboard controller for full robot control."""
        print("=== Setting up TeleMoMa keyboard controller ===")

        if self.robot is None:
            raise RuntimeError("Cannot setup controller without a robot")

        # Configure TeleMoMa for keyboard control
        # Similar to robot_teleoperate_demo.py
        teleop_config.arm_left_controller = "keyboard"
        teleop_config.arm_right_controller = "keyboard"
        teleop_config.base_controller = "keyboard"

        # Set speed parameters for comfortable control
        # arm_speed_scaledown: controls how fast the end effector moves
        # Lower value = slower, more precise control
        # Recommended: 0.02-0.08 (0.04 is a good default)
        teleop_config.interface_kwargs["keyboard"] = {
            "arm_speed_scaledown": 0.04,  # Slow down arm movement for precision
        }

        print("  Left arm: Keyboard control")
        print("  Right arm: Keyboard control")
        print("  Base: Keyboard control (WASD)")

        # Initialize TeleopSystem
        self.teleop_sys = TeleopSystem(config=teleop_config, robot=self.robot, show_control_marker=self.show_marker)

        # Start the teleoperation system
        self.teleop_sys.start()

        print("\n✓ Keyboard controller ready!")
        self._print_controls()

    def _print_controls(self):
        """Print keyboard control instructions."""
        print("\n" + "=" * 80)
        print("KEYBOARD CONTROLS (TeleMoMa)")
        print("=" * 80)
        print("\nLEFT ARM (End Effector Position):")
        print("  W/S       : Forward/backward")
        print("  A/D       : Left/right")
        print("  Q/E       : Up/down")
        print("  X/C, T/B, Z/V : Orientation control")
        print("  4         : Toggle left gripper")
        print("\nRIGHT ARM (End Effector Position):")
        print("  I/K       : Forward/backward")
        print("  J/L       : Left/right")
        print("  U/O       : Up/down")
        print("  M/,, P/;, N/. : Orientation control")
        print("  7         : Toggle right gripper")
        print("\nBASE MOVEMENT:")
        print("  Arrow Up/Down    : Forward/backward")
        print("  Arrow Left/Right : Turn left/right")
        print("  [/]              : Strafe left/right")
        print("\nOTHER:")
        print("  -/+       : Torso up/down")
        print("  ESC       : Quit (GUI mode)")
        print("  Ctrl+C    : Quit (always works)")
        print("\nWARNING:")
        print("  WASD controls LEFT ARM, not base!")
        print("  Use arrow keys to move the base")
        print("=" * 80 + "\n")

    @property
    def video_writer(self) -> Tuple[Container, Stream]:
        """Returns the video writer for the current teleoperation session."""
        return self._video_writer

    @video_writer.setter
    def video_writer(self, video_writer: Tuple[Container, Stream]) -> None:
        """Sets the video writer and closes the previous one if exists."""
        if self._video_writer is not None:
            (container, stream) = self._video_writer
            # Flush any remaining packets
            for packet in stream.encode():
                container.mux(packet)
            # Close the container
            container.close()
        self._video_writer = video_writer

    def _preprocess_obs(self, obs: dict) -> dict:
        """
        Preprocess the observation dictionary.

        Args:
            obs: The observation dictionary to preprocess.

        Returns:
            dict: The preprocessed observation dictionary.
        """
        import torch as th

        obs = flatten_obs_dict(obs)
        base_pose = self.robot.get_position_orientation()
        cam_rel_poses = []

        # Get the actual robot name (e.g., "robot_seemfy", not "robot_r1")
        robot_name = self.robot.name

        # Get camera relative poses
        # The sensor name format is: "{robot_name}:{link_name}:Camera:0"
        for camera_id, camera_template in ROBOT_CAMERA_NAMES["R1Pro"].items():
            # camera_template format: "robot_r1::robot_r1:left_realsense_link:Camera:0"
            # Split by ":" -> ['robot_r1', '', 'robot_r1', 'left_realsense_link', 'Camera', '0']
            # We need indices [3:] to get ['left_realsense_link', 'Camera', '0']
            link_part = camera_template.split(":")[3:]  # Skip 'robot_r1', '', 'robot_r1'
            link_name = ":".join(link_part)  # 'left_realsense_link:Camera:0'

            # Build the actual sensor key with the real robot name
            sensor_key = f"{robot_name}:{link_name}"

            # Get the sensor
            if sensor_key not in self.robot.sensors:
                print(
                    f"Camera '{camera_id}' (key: '{sensor_key}') not found. Available: {list(self.robot.sensors.keys())}"
                )
                continue

            camera = self.robot.sensors[sensor_key]
            direct_cam_pose = camera.camera_parameters["cameraViewTransform"]
            if np.allclose(direct_cam_pose, np.zeros(16)):
                cam_rel_poses.append(
                    th.cat(T.relative_pose_transform(*(camera.get_position_orientation()), *base_pose))
                )
            else:
                cam_pose = T.mat2pose(th.tensor(np.linalg.inv(np.reshape(direct_cam_pose, [4, 4]).T), dtype=th.float32))
                cam_rel_poses.append(th.cat(T.relative_pose_transform(*cam_pose, *base_pose)))

        # Use the standard key for compatibility with BEHAVIOR-1K datasets
        if len(cam_rel_poses) > 0:
            obs["robot_r1::cam_rel_poses"] = th.cat(cam_rel_poses, axis=-1)

        return obs

    def _write_video(self) -> None:
        """Write the current robot observations to video."""
        if not self.write_video or self.video_writer is None:
            return

        try:
            # Get the actual robot name for building observation keys
            robot_name = self.robot.name

            # Build observation keys based on actual robot name
            # Observation keys format: "{robot_name}::{robot_name}:{link_name}:Camera:0::rgb"
            def get_camera_obs_key(camera_id):
                """Build the correct observation key for a camera."""
                camera_template = ROBOT_CAMERA_NAMES["R1Pro"][camera_id]
                # Extract link part: "left_realsense_link:Camera:0"
                # Split by ":" -> ['robot_r1', '', 'robot_r1', 'left_realsense_link', 'Camera', '0']
                # We need indices [3:] to get ['left_realsense_link', 'Camera', '0']
                link_part = camera_template.split(":")[3:]  # Skip 'robot_r1', '', 'robot_r1'
                link_name = ":".join(link_part)
                # Build key: "robot_fqjdnf::robot_fqjdnf:left_realsense_link:Camera:0::rgb"
                return f"{robot_name}::{robot_name}:{link_name}::rgb"

            left_wrist_key = get_camera_obs_key("left_wrist")
            right_wrist_key = get_camera_obs_key("right_wrist")
            head_key = get_camera_obs_key("head")

            # Check if keys exist
            if not all(key in self.obs for key in [left_wrist_key, right_wrist_key, head_key]):
                print(f"Missing camera observations. Available keys: {list(self.obs.keys())[:10]}...")
                return

            # Concatenate observations from different cameras
            left_wrist_rgb = cv2.resize(
                self.obs[left_wrist_key].numpy(),
                (224, 224),
            )
            right_wrist_rgb = cv2.resize(
                self.obs[right_wrist_key].numpy(),
                (224, 224),
            )
            head_rgb = cv2.resize(
                self.obs[head_key].numpy(),
                (448, 448),
            )

            # Stack images: left and right wrist on the left, head camera on the right
            frame = np.hstack([np.vstack([left_wrist_rgb, right_wrist_rgb]), head_rgb])

            # Write to file
            write_video(
                np.expand_dims(frame, 0),
                video_writer=self.video_writer,
                batch_size=1,
                mode="rgb",
            )

            # Push to live stream
            if self.enable_streaming and self.video_streamer is not None:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                self.video_streamer.push_frame(frame_bgr)
        except Exception as e:
            print(f"Error writing video frame: {e}")
            # Don't crash, just skip this frame

    def run(self):
        """Main teleoperation loop with full robot control."""
        try:
            # Setup environment and controller
            self.setup_environment()
            self.setup_keyboard_controller()

            # Start video streaming server if enabled
            if self.enable_streaming:
                self.video_streamer = VideoStreamer(port=self.stream_port)
                self.video_streamer.start()

            # Create video directory if needed
            if self.write_video:
                video_path = self.log_path / "videos"
                video_path.mkdir(parents=True, exist_ok=True)
                print(f"\nVideo recording enabled. Videos will be saved to: {video_path}")

            # Reset environment
            print("Resetting environment...")
            obs_raw, _ = self.env.reset()
            self.obs = self._preprocess_obs(obs_raw)

            # Set up viewer camera if not headless
            if not self.headless:
                print("Setting up viewer camera...")
                # Position camera for good view of the robot
                og.sim.viewer_camera.set_position_orientation(
                    position=[2.0, 2.0, 1.5], orientation=[0.0, 0.3, 0.7, 0.6]
                )
                # Enable camera teleoperation (can move with mouse)
                og.sim.enable_viewer_camera_teleoperation()

            print("\n" + "=" * 80)
            print("TELEOPERATION STARTED - Control the robot with your keyboard!")
            print("=" * 80)
            print(f"Task: {self.task_name}")
            print(f"Press ESC to quit")
            print("=" * 80 + "\n")

            step_count = 0
            episode_count = 0

            # Create video writer for first episode
            if self.write_video:
                video_name = str(video_path / f"{self.task_name}_episode_{episode_count}.mp4")
                self.video_writer = create_video_writer(
                    fpath=video_name,
                    resolution=(448, 672),
                )
                print(f"Start Recording to: {video_name}...\n")

            # Main simulation loop
            while True:
                # Get action from teleoperation system
                # This reads keyboard input and converts to robot action
                action = self.teleop_sys.get_action(self.teleop_sys.get_obs())

                # Step the environment
                obs_raw, reward, terminated, truncated, info = self.env.step(action)
                self.obs = self._preprocess_obs(obs_raw)
                step_count += 1

                # Write video frame
                if self.write_video:
                    self._write_video()

                # Print status occasionally
                if step_count % 500 == 0:
                    print(f"[Step {step_count}] Running... (Press ESC to quit)")

                # Check if episode ended
                done = terminated or truncated
                if done:
                    success = info.get("done", {}).get("success", False)

                    # Close current video
                    if self.write_video:
                        self.video_writer = None
                        print(f"\n✅ Video saved: {video_name}")

                    print("\n" + "=" * 80)
                    print(f"������ EPISODE {episode_count} COMPLETED")
                    print("=" * 80)
                    print(f"  Steps: {step_count}")
                    print(f"  Success: {'✓ YES' if success else '✗ NO'}")
                    print(f"  Reward: {reward:.3f}")
                    if terminated:
                        print(f"  Reason: Task completed")
                    elif truncated:
                        print(f"  Reason: Time limit reached")
                    print("=" * 80)

                    # Ask user if they want to continue
                    print("\nResetting environment for next episode...")
                    print("(Press ESC during simulation to quit)\n")

                    obs_raw, _ = self.env.reset()
                    self.obs = self._preprocess_obs(obs_raw)
                    step_count = 0
                    episode_count += 1

                    # Create new video for next episode
                    if self.write_video:
                        video_name = str(video_path / f"{self.task_name}_episode_{episode_count}.mp4")
                        self.video_writer = create_video_writer(
                            fpath=video_name,
                            resolution=(448, 672),
                        )
                        print(f"Start Recording to: {video_name}...\n")

        except KeyboardInterrupt:
            print("\n\nInterrupted by user (Ctrl+C)")
        except Exception as e:
            print(f"\n\nError occurred: {e}")
            import traceback

            traceback.print_exc()
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        print("\n" + "=" * 80)
        print("Cleaning up...")
        print("=" * 80)

        # Stop video streamer
        if self.video_streamer is not None:
            try:
                self.video_streamer.stop()
                print("✓ Video streamer stopped")
            except Exception as e:
                print(f"Error stopping video streamer: {e}")

        # Close video writer
        if self.write_video and self._video_writer is not None:
            try:
                self.video_writer = None
                print("✓ Video writer closed")
            except Exception as e:
                print(f"Error closing video writer: {e}")

        # Stop teleoperation system
        if self.teleop_sys:
            try:
                self.teleop_sys.stop()
                print("✓ Teleoperation system stopped")
            except Exception as e:
                print(f"Error stopping teleop system: {e}")

        # Close environment
        if self.env:
            try:
                self.env.close()
                print("✓ Environment closed")
            except Exception as e:
                print(f"Error closing environment: {e}")

        # Clear OmniGibson
        try:
            og.clear()
            print("✓ OmniGibson cleared")
        except Exception as e:
            print(f"Error clearing OmniGibson: {e}")

        print("=" * 80)
        print("Cleanup complete!")
        print("=" * 80 + "\n")


def main():
    """Main entry point for BEHAVIOR-1K teleoperation."""
    parser = argparse.ArgumentParser(
        description="Local keyboard teleoperation for OmniGibson BEHAVIOR-1K tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with GUI (recommended for beginners)
  python teleop_local.py --task turning_on_radio
  
  # Run with control markers visible
  python teleop_local.py --task picking_up_trash --show-marker
  
  # Run headless (no visualization, for remote servers)
  python teleop_local.py --task freeze_pies --headless
  
  # Specify custom BEHAVIOR directory and max steps
  python teleop_local.py --task turning_on_radio \\
      --behavior-base /path/to/behavior \\
      --max-steps 1000

Available tasks (examples):
  - turning_on_radio
  - picking_up_trash
  - freeze_pies
  - putting_away_Halloween_decorations
  - cleaning_up_plates_and_food
  - can_meat
  - setting_mousetraps
  - (see openpi/scripts/submit_openpi_eval.sh for full list)

Controls:
  - WASD: Robot base movement
  - Arrow keys, I/K/J/L: Arm control
  - [/], ;/': Gripper control
  - ESC: Quit
  - See on-screen instructions for full control list
        """,
    )

    parser.add_argument(
        "--task",
        type=str,
        default="turning_on_radio",
        help="Name of the BEHAVIOR task to run (default: turning_on_radio)",
    )

    parser.add_argument("--headless", action="store_true", help="Run in headless mode (no GUI)")

    parser.add_argument(
        "--behavior-base",
        type=str,
        default=None,
        help="Base directory for BEHAVIOR (default: from BEHAVIOR_BASE env var or hardcoded path)",
    )

    parser.add_argument(
        "--show-marker",
        action="store_true",
        default=True,
        help="Show green control markers for end effectors (default: True)",
    )

    parser.add_argument("--no-marker", action="store_false", dest="show_marker", help="Hide control markers")

    parser.add_argument(
        "--max-steps", type=int, default=None, help="Maximum steps per episode (default: None, uses task default)"
    )

    parser.add_argument(
        "--write-video",
        action="store_true",
        default=True,
        help="Record video of the teleoperation session (default: True)",
    )

    parser.add_argument("--no-video", action="store_false", dest="write_video", help="Disable video recording")

    parser.add_argument(
        "--log-path",
        type=str,
        default=None,
        help="Path to save videos and logs (default: behavior_base/outputs/teleop_videos)",
    )

    parser.add_argument(
        "--stream-port",
        type=int,
        default=5000,
        help="Port for live video streaming (default: 5000, enabled in headless mode)",
    )

    args = parser.parse_args()

    # Print banner
    print("\n" + "=" * 80)
    print("BEHAVIOR-1K Keyboard Teleoperation")
    print("=" * 80)
    print("Using TeleMoMa keyboard interface for full robot control")
    print("Based on OmniGibson's robot_teleoperate_demo.py")
    print("=" * 80 + "\n")

    # Create and run teleop controller
    teleop = TeleopController(
        task_name=args.task,
        headless=args.headless,
        behavior_base=args.behavior_base,
        show_marker=args.show_marker,
        max_steps=args.max_steps,
        write_video=args.write_video,
        log_path=args.log_path,
        stream_port=args.stream_port,
    )

    teleop.run()


if __name__ == "__main__":
    main()
