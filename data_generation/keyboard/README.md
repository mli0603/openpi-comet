# BEHAVIOR-1K Keyboard Teleoperation Script

Local keyboard teleoperation script for OmniGibson BEHAVIOR-1K tasks with full arm and base control.

## Features

- 🎮 **Keyboard Teleoperation**: Control R1Pro robot using TeleMoMa keyboard interface
- 📹 **Video Recording**: Automatically record teleoperation sessions to MP4 files
- 🌐 **Live Video Streaming**: Real-time browser-based viewing via Flask/MJPEG
- 🖥️ **Headless Mode**: Run on remote servers without GUI
- 🎯 **Multi-task Support**: Compatible with all BEHAVIOR-1K tasks

## Dependencies

- OmniGibson
- TeleMoMa (`pip install telemoma`)
- Flask
- OpenCV
- NumPy

## Usage

### Basic Commands

```bash
python teleop.py --headless --task turning_on_radio --behavior-base /home/b1k --max-steps 2000
```

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--task` | `turning_on_radio` | Name of the BEHAVIOR task |
| `--headless` | `False` | Run without GUI |
| `--behavior-base` | env var or default path | Base directory for BEHAVIOR |
| `--show-marker` | `True` | Show end-effector control markers |
| `--no-marker` | - | Hide control markers |
| `--max-steps` | `None` | Maximum steps per episode |
| `--write-video` | `True` | Enable video recording |
| `--no-video` | - | Disable video recording |
| `--log-path` | `behavior_base/outputs/teleop_videos` | Video output path |
| `--stream-port` | `5000` | Port for live video streaming |

## Keyboard Controls

### Left Arm
| Key | Action |
|-----|--------|
| `W` / `S` | Forward / Backward |
| `A` / `D` | Left / Right |
| `Q` / `E` | Up / Down |
| `X`/`C`, `T`/`B`, `Z`/`V` | Orientation control |
| `4` | Toggle left gripper |

### Right Arm
| Key | Action |
|-----|--------|
| `I` / `K` | Forward / Backward |
| `J` / `L` | Left / Right |
| `U` / `O` | Up / Down |
| `M`/`,`, `P`/`;`, `N`/`.` | Orientation control |
| `7` | Toggle right gripper |

### Base Movement
| Key | Action |
|-----|--------|
| `↑` / `↓` | Forward / Backward |
| `←` / `→` | Turn Left / Turn Right |
| `[` / `]` | Strafe Left / Strafe Right |

### Other Controls
| Key | Action |
|-----|--------|
| `-` / `+` | Torso Down / Up |
| `ESC` | Quit (GUI mode) |
| `Ctrl+C` | Quit (always works) |

> ⚠️ **Warning**: WASD controls the **left arm**, not the base! Use **arrow keys** to move the base.

## Live Video Streaming

After starting the script, access the live video stream in your browser:

```
http://localhost:5000
```

The stream displays three camera views: left wrist, right wrist, and head camera.

## Available Tasks (Examples)

- `turning_on_radio`
- `picking_up_trash`
- `freeze_pies`
- `putting_away_Halloween_decorations`
- `cleaning_up_plates_and_food`
- `can_meat`
- `setting_mousetraps`

## Output Files

Video files are saved to `{log_path}/videos/` with the naming format:
```
{task_name}_episode_{episode_number}.mp4
```

## Notes

1. Ensure OmniGibson and TeleMoMa are properly installed
2. Use `--headless` mode when running on remote servers
3. Video streaming requires network access
4. Environment automatically resets after each episode with new recording
