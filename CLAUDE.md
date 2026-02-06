# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TWIST2 Sim2Sim controller for the Unitree G1 humanoid robot in NVIDIA Isaac Sim. Runs trained ONNX neural network policies on simulated robots, communicating with an external policy server via Redis.

## Build & Run Commands

```bash
# Install (editable mode, from Isaac Sim conda environment)
pip install -e .

# Run simulation (requires Redis server running on localhost:6379)
python scripts/run_sim2sim.py --robot g1 --policy assets/ckpts/twist2_1017_20k.onnx --device cuda

# Headless mode
python scripts/run_sim2sim.py --headless

# Key CLI arguments
#   --robot        g1 | g1_inspire (robot variant)
#   --control-mode torque | pd (default: pd)
#   --policy       path to ONNX checkpoint
#   --device       cuda | cpu
#   --physics_dt   physics timestep (default: 0.002 = 500 Hz)
#   --policy_frequency  policy rate (default: 100 Hz)
#   --headless     run without rendering
```

No test suite or linting tools are currently configured.

## Architecture

### Two-Rate Control Loop
- **Policy loop (100 Hz):** Collects proprioceptive observations, runs ONNX inference, publishes state to Redis
- **Servo loop (500 Hz):** Applies PD control every physics step — either explicit torque computation ("torque" mode) or PhysX implicit targets ("pd" mode)
- Decimation ratio (servo/policy) determined by `physics_dt * policy_frequency`

### Key Components
- **`isaac_twist2/g1_twist2_controller.py`** — Core controller class (`G1Twist2Controller`). Manages robot articulation, observation construction, ONNX policy inference, and PD control. ~390 LOC, contains all robot-specific constants (joint mappings, PD gains, torque limits, observation dimensions).
- **`isaac_twist2/rot_utils.py`** — Quaternion-to-Euler conversion (Isaac Sim's scalar-first convention: `[qw, qx, qy, qz]`)
- **`scripts/run_sim2sim.py`** — Entry point. Sets up Isaac Sim world, instantiates controller, runs physics callback loop.

### Observation Vector
92-dim proprioceptive state per timestep: angular velocity (3) + roll/pitch (2) + joint positions relative to default (29) + joint velocities (29) + last action (29). Stacked with 10-frame history + 35-dim mimic observation = 1432 total dimensions.

### Redis Protocol
Bidirectional state exchange at policy rate. Sends body/hand/neck joint states, receives mimic actions. Uses Redis pipelines for low-latency batched reads/writes.

### Robot Configuration
29 controllable DOFs: 12 leg joints + 3 waist joints + 14 arm/wrist joints. PD gains (stiffness/damping) and torque limits are per-joint constants defined in the controller.

## Assets
- `assets/ckpts/` — ONNX policy checkpoints (twist2_1017_20k.onnx, twist2_1017_25k.onnx)
- `assets/g1_usd/` — Standard G1 robot USD model
- `assets/g1_inspire_hand_usd/` — G1 with Inspire dexterous hand variant

## Dependencies
- Isaac Sim SDK (`isaacsim.core.api`)
- `redis` — inter-process communication
- `onnxruntime-gpu` — policy inference
- `numpy`, `torch`
- Requires CUDA-capable GPU and running Redis server
