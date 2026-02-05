#!/usr/bin/env python3
"""Standalone Isaac Sim runner for TWIST2 G1 sim2sim.

Usage:
    isaacsim-python scripts/run_sim2sim.py [--policy PATH] [--usd PATH] [--device cuda|cpu]
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="TWIST2 G1 Sim2Sim in Isaac Sim")
    parser.add_argument(
        "--policy",
        type=str,
        default="assets/ckpts/twist2_1017_20k.onnx",
        help="Path to ONNX policy file",
    )
    parser.add_argument(
        "--usd",
        type=str,
        default="assets/g1_usd/g1_29dof_inspire_hand.usd",
        help="Path to G1 USD file",
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--physics_dt", type=float, default=0.002, help="Physics timestep (500 Hz)")
    parser.add_argument("--policy_frequency", type=int, default=100, help="Policy frequency in Hz")
    parser.add_argument("--headless", action="store_true", help="Run without rendering")
    args = parser.parse_args()

    # Resolve paths relative to project root (where this script is run from)
    policy_path = os.path.abspath(args.policy)
    usd_path = os.path.abspath(args.usd)

    if not os.path.exists(policy_path):
        print(f"Error: Policy file not found: {policy_path}")
        sys.exit(1)
    if not os.path.exists(usd_path):
        print(f"Error: USD file not found: {usd_path}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Isaac Sim must be initialized before any Omniverse imports
    # ------------------------------------------------------------------
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": args.headless})

    import numpy as np
    from isaacsim.core.api import World

    from isaac_twist2 import G1Twist2Controller

    # ------------------------------------------------------------------
    # World setup
    # ------------------------------------------------------------------
    physics_dt = args.physics_dt
    decimation = int(1.0 / (args.policy_frequency * physics_dt))
    rendering_dt = 5 * physics_dt  # render at 100 Hz

    print(f"[TWIST2] Physics dt: {physics_dt}s ({1/physics_dt:.0f} Hz)")
    print(f"[TWIST2] Policy frequency: {args.policy_frequency} Hz (decimation={decimation})")
    print(f"[TWIST2] Rendering dt: {rendering_dt}s ({1/rendering_dt:.0f} Hz)")

    world = World(
        stage_units_in_meters=1.0,
        physics_dt=physics_dt,
        rendering_dt=rendering_dt,
    )

    world.scene.add_default_ground_plane()

    # ------------------------------------------------------------------
    # Spawn controller
    # ------------------------------------------------------------------
    controller = G1Twist2Controller(
        prim_path="/World/g1",
        usd_path=usd_path,
        policy_path=policy_path,
        device=args.device,
        position=np.array([0.0, 0.0, 0.793]),
    )

    # reset() does synchronous init: initializes physics, starts timeline, steps once
    world.reset()

    controller.initialize()
    controller.post_reset()

    # ------------------------------------------------------------------
    # Physics callback — two-rate control loop
    # ------------------------------------------------------------------
    step_counter = [0]  # mutable container for closure

    def on_physics_step(step_size):
        if step_counter[0] % decimation == 0:
            controller._run_policy()
        controller.apply_torques()
        step_counter[0] += 1

    world.add_physics_callback("sim_step", on_physics_step)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    print("[TWIST2] Entering main loop. Close the window or Ctrl-C to stop.")
    try:
        while simulation_app.is_running():
            world.step(render=True)
    except KeyboardInterrupt:
        print("[TWIST2] Interrupted.")

    simulation_app.close()


if __name__ == "__main__":
    main()
