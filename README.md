# isaac_twist2

TWIST2 Sim2Sim controller for the [Unitree G1](https://www.unitree.com/g1/) humanoid robot in [NVIDIA Isaac Sim](https://developer.nvidia.com/isaac-sim). Runs trained ONNX neural network policies in simulation, communicating with an external motion server via Redis for real-time mimic control.

> **Note:** This project is designed to run alongside the [TWIST2](https://github.com/amazon-far/TWIST2) repo, which provides the motion server that communicates with this controller via Redis.

> **Note:** Currently only PD control mode (`--control-mode pd`) is supported.

## Prerequisites

- NVIDIA Isaac Sim (with `isaacsim.core.api`)
- Python 3.11+
- CUDA-capable GPU
- Redis server running on `localhost:6379`

## Installation

```bash
# From within your Isaac Sim conda environment
pip install -e .
```

## Usage

Start the Redis server, then run the simulation:

```bash
python scripts/run_sim2sim.py
```

### Command-Line Options

| Flag | Default | Description |
|---|---|---|
| `--robot` | `g1` | Robot variant: `g1` or `g1_inspire` |
| `--policy` | `assets/ckpts/twist2_1017_20k.onnx` | Path to ONNX policy checkpoint |
| `--usd` | (derived from `--robot`) | Override USD model path |
| `--device` | `cuda` | Inference device: `cuda` or `cpu` |
| `--control-mode` | `pd` | `pd` (PhysX implicit PD) or `torque` (explicit PD) |
| `--physics_dt` | `0.002` | Physics timestep in seconds (500 Hz) |
| `--policy_frequency` | `100` | Policy inference rate in Hz |
| `--headless` | off | Run without rendering |

### Examples

```bash
# G1 with Inspire hand, explicit torque control
python scripts/run_sim2sim.py --robot g1_inspire --control-mode torque

# Headless with CPU inference
python scripts/run_sim2sim.py --headless --device cpu

# Custom policy checkpoint
python scripts/run_sim2sim.py --policy assets/ckpts/twist2_1017_25k.onnx
```

## Architecture

### Two-Rate Control Loop

The controller runs at two frequencies:

- **Policy (100 Hz):** Reads joint states, constructs a 1432-dim observation vector (proprioception + 10-frame history + mimic actions), runs ONNX inference, and publishes state to Redis.
- **Servo (500 Hz):** Applies PD control every physics step. In `torque` mode, computes explicit PD torques. In `pd` mode, sends position targets to PhysX's built-in PD solver.

### Redis Protocol

The controller exchanges state and actions with an external motion server over Redis:

- **Publishes:** `state_body_*`, `state_hand_left_*`, `state_hand_right_*`, `state_neck_*`, `t_state`
- **Subscribes:** `action_body_*`, `action_hand_left_*`, `action_hand_right_*`, `action_neck_*`

All values are JSON-encoded float arrays.

### Robot Configuration

29 controllable body DOFs:

| Group | Joints |
|---|---|
| Left leg | hip pitch/roll/yaw, knee, ankle pitch/roll |
| Right leg | hip pitch/roll/yaw, knee, ankle pitch/roll |
| Torso | waist yaw/roll/pitch |
| Left arm | shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw |
| Right arm | shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw |

### Supported Robot Variants

| Variant | USD Path |
|---|---|
| `g1` | `assets/g1_usd/g1.usd` |
| `g1_inspire` | `assets/g1_usd/g1_inspire.usd` |

## Project Structure

```
isaac_twist2/
├── isaac_twist2/
│   ├── g1_twist2_controller.py   # Core controller (policy, PD control, Redis I/O)
│   └── rot_utils.py              # Quaternion-to-Euler conversion
├── scripts/
│   └── run_sim2sim.py            # Isaac Sim entry point
├── assets/
│   ├── ckpts/                    # ONNX policy checkpoints
│   ├── g1_usd/                   # G1 robot model
└── config/
    └── extension.toml            # Isaac Sim extension metadata
```

## License

MIT
