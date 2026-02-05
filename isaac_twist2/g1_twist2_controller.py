import json
import time
from collections import deque

import numpy as np
import onnxruntime as ort
import redis
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.prims import define_prim, get_prim_at_path
from isaacsim.core.utils.rotations import quat_to_rot_matrix

from .rot_utils import quatToEuler

# ---------------------------------------------------------------------------
# Constants — match server_low_level_g1_sim.py exactly
# ---------------------------------------------------------------------------
NUM_BODY_DOFS = 29
N_MIMIC_OBS = 35
N_PROPRIO = 92  # 3 + 2 + 3*29
N_OBS_SINGLE = 127  # N_MIMIC_OBS + N_PROPRIO
HISTORY_LEN = 10
TOTAL_OBS_SIZE = 1432  # N_OBS_SINGLE * (HISTORY_LEN + 1) + N_MIMIC_OBS

ACTION_SCALE = 0.5
ANG_VEL_SCALE = 0.25
DOF_VEL_SCALE = 0.05
ANKLE_IDX = [4, 5, 10, 11]

BODY_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

DEFAULT_DOF_POS = np.array([
    -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,        # left leg
    -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,        # right leg
     0.0, 0.0, 0.0,                          # torso
     0.0, 0.4, 0.0, 1.2, 0.0, 0.0, 0.0,    # left arm
     0.0,-0.4, 0.0, 1.2, 0.0, 0.0, 0.0,    # right arm
], dtype=np.float32)

STIFFNESS = np.array([
    100, 100, 100, 150, 40, 40,
    100, 100, 100, 150, 40, 40,
    150, 150, 150,
    40, 40, 40, 40, 4.0, 4.0, 4.0,
    40, 40, 40, 40, 4.0, 4.0, 4.0,
], dtype=np.float32)

DAMPING = np.array([
    2, 2, 2, 4, 2, 2,
    2, 2, 2, 4, 2, 2,
    4, 4, 4,
    5, 5, 5, 5, 0.2, 0.2, 0.2,
    5, 5, 5, 5, 0.2, 0.2, 0.2,
], dtype=np.float32)

TORQUE_LIMITS = np.array([
    100, 100, 100, 150, 40, 40,
    100, 100, 100, 150, 40, 40,
    150, 150, 150,
    40, 40, 40, 40, 4.0, 4.0, 4.0,
    40, 40, 40, 40, 4.0, 4.0, 4.0,
], dtype=np.float32)

# Redis keys (matching MuJoCo sim2sim)
_REDIS_STATE_BODY = "state_body_unitree_g1_with_hands"
_REDIS_STATE_HAND_L = "state_hand_left_unitree_g1_with_hands"
_REDIS_STATE_HAND_R = "state_hand_right_unitree_g1_with_hands"
_REDIS_STATE_NECK = "state_neck_unitree_g1_with_hands"
_REDIS_T_STATE = "t_state"
_REDIS_ACTION_BODY = "action_body_unitree_g1_with_hands"
_REDIS_ACTION_HAND_L = "action_hand_left_unitree_g1_with_hands"
_REDIS_ACTION_HAND_R = "action_hand_right_unitree_g1_with_hands"
_REDIS_ACTION_NECK = "action_neck_unitree_g1_with_hands"


class G1Twist2Controller:
    """TWIST2 controller for the G1 robot in Isaac Sim.

    Manages its own SingleArticulation. Runs ONNX policy at 100 Hz and
    explicit PD torque servo at 500 Hz, communicating with the motion
    server via Redis.
    """

    def __init__(
        self,
        prim_path: str,
        usd_path: str,
        policy_path: str,
        device: str = "cuda",
        position: np.ndarray | None = None,
        orientation: np.ndarray | None = None,
    ):
        # -- Create prim and articulation --------------------------------
        prim = get_prim_at_path(prim_path)
        if not prim.IsValid():
            prim = define_prim(prim_path, "Xform")
            prim.GetReferences().AddReference(usd_path)

        self.robot = SingleArticulation(
            prim_path=prim_path,
            name="g1_twist2",
            position=position,
            orientation=orientation,
        )

        # -- Load ONNX policy -------------------------------------------
        if device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(policy_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        print(f"[TWIST2] ONNX policy loaded from {policy_path}")
        print(f"[TWIST2]   providers: {self.session.get_providers()}")

        # -- Redis -------------------------------------------------------
        self.redis_client = redis.Redis(host="localhost", port=6379, db=0)
        self.redis_pipeline = self.redis_client.pipeline()

        # -- State buffers -----------------------------------------------
        self.proprio_history_buf: deque = deque(maxlen=HISTORY_LEN)
        for _ in range(HISTORY_LEN):
            self.proprio_history_buf.append(np.zeros(N_OBS_SINGLE, dtype=np.float32))

        self.last_action = np.zeros(NUM_BODY_DOFS, dtype=np.float32)
        self.pd_target = DEFAULT_DOF_POS.copy()
        self.policy_counter = 0

        # FPS tracking
        self._last_policy_time: float | None = None
        self._policy_intervals: list[float] = []
        self._policy_step_count = 0

        # Filled during initialize()
        self.body_dof_indices: np.ndarray | None = None
        self.hand_dof_indices: np.ndarray | None = None
        self._num_dof: int = 0

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def initialize(self, physics_sim_view=None):
        self.robot.initialize(physics_sim_view=physics_sim_view)
        self._num_dof = self.robot.num_dof

        # Set effort / control mode
        self.robot.get_articulation_controller().set_effort_modes("force")
        self.robot.get_articulation_controller().switch_control_mode("effort")

        # -- Build joint mapping -----------------------------------------
        dof_names = list(self.robot.dof_names)
        print(f"[TWIST2] Articulation has {self._num_dof} DOFs: {dof_names}")

        body_indices = []
        for name in BODY_JOINT_NAMES:
            idx = self._find_joint_index(name, dof_names)
            if idx is None:
                raise RuntimeError(
                    f"[TWIST2] Could not find body joint '{name}' in articulation DOFs.\n"
                    f"  Expected: {BODY_JOINT_NAMES}\n"
                    f"  Actual:   {dof_names}"
                )
            body_indices.append(idx)

        self.body_dof_indices = np.array(body_indices, dtype=np.int64)
        all_indices = set(range(self._num_dof))
        self.hand_dof_indices = np.array(
            sorted(all_indices - set(body_indices)), dtype=np.int64
        )
        print(f"[TWIST2] Body DOF indices ({len(self.body_dof_indices)}): {self.body_dof_indices}")
        print(f"[TWIST2] Hand DOF indices ({len(self.hand_dof_indices)}): {self.hand_dof_indices}")

        # -- Build full-articulation default arrays ----------------------
        self.full_default_pos = np.zeros(self._num_dof, dtype=np.float32)
        self.full_default_pos[self.body_dof_indices] = DEFAULT_DOF_POS

        # -- Disable Isaac Sim internal PD (we compute torques ourselves) --
        zero_gains = np.zeros(self._num_dof, dtype=np.float32)
        self.robot._articulation_view.set_gains(zero_gains, zero_gains)

        # Set high max efforts so our explicit torques aren't clamped
        high_effort = np.full(self._num_dof, 1000.0, dtype=np.float32)
        self.robot._articulation_view.set_max_efforts(high_effort)

        # Publish initial state to Redis
        self.redis_pipeline.set(_REDIS_STATE_BODY, json.dumps(np.zeros(34).tolist()))
        self.redis_pipeline.set(_REDIS_STATE_HAND_L, json.dumps(np.zeros(7).tolist()))
        self.redis_pipeline.set(_REDIS_STATE_HAND_R, json.dumps(np.zeros(7).tolist()))
        self.redis_pipeline.set(_REDIS_STATE_NECK, json.dumps(np.zeros(2).tolist()))
        self.redis_pipeline.execute()

    # ------------------------------------------------------------------
    # Policy step (100 Hz)
    # ------------------------------------------------------------------
    def _run_policy(self):
        # -- Read robot state --------------------------------------------
        joint_positions = self.robot.get_joint_positions()
        joint_velocities = self.robot.get_joint_velocities()
        body_pos = joint_positions[self.body_dof_indices]
        body_vel = joint_velocities[self.body_dof_indices]

        # World pose & angular velocity → body frame
        _, quat_world = self.robot.get_world_pose()
        ang_vel_world = self.robot.get_angular_velocity()
        R_IB = quat_to_rot_matrix(quat_world)
        R_BI = R_IB.T
        ang_vel_body = R_BI @ ang_vel_world

        rpy = quatToEuler(quat_world)  # [roll, pitch, yaw]

        # Zero ankle velocities (match MuJoCo)
        obs_body_vel = body_vel.copy()
        obs_body_vel[ANKLE_IDX] = 0.0

        # -- Build obs_proprio (92) --------------------------------------
        obs_proprio = np.concatenate([
            ang_vel_body * ANG_VEL_SCALE,            # 3
            rpy[:2],                                   # 2
            (body_pos - DEFAULT_DOF_POS),              # 29
            obs_body_vel * DOF_VEL_SCALE,              # 29
            self.last_action,                          # 29
        ]).astype(np.float32)

        # -- Build state_body (34) and publish to Redis -------------------
        state_body = np.concatenate([
            ang_vel_body,   # 3
            rpy[:2],        # 2
            body_pos,       # 29
        ]).astype(np.float32)

        self.redis_pipeline.set(_REDIS_STATE_BODY, json.dumps(state_body.tolist()))
        self.redis_pipeline.set(_REDIS_STATE_HAND_L, json.dumps(np.zeros(7).tolist()))
        self.redis_pipeline.set(_REDIS_STATE_HAND_R, json.dumps(np.zeros(7).tolist()))
        self.redis_pipeline.set(_REDIS_STATE_NECK, json.dumps(np.zeros(2).tolist()))
        self.redis_pipeline.set(_REDIS_T_STATE, int(time.time() * 1000))
        self.redis_pipeline.execute()

        # -- Read mimic actions from Redis --------------------------------
        for key in [_REDIS_ACTION_BODY, _REDIS_ACTION_HAND_L, _REDIS_ACTION_HAND_R, _REDIS_ACTION_NECK]:
            self.redis_pipeline.get(key)
        redis_results = self.redis_pipeline.execute()

        action_mimic = np.array(json.loads(redis_results[0]), dtype=np.float32) if redis_results[0] else np.zeros(N_MIMIC_OBS, dtype=np.float32)
        # action_hand_left, action_hand_right, action_neck not used for policy obs but consumed

        # -- Build full observation (1432) --------------------------------
        obs_full = np.concatenate([action_mimic, obs_proprio]).astype(np.float32)  # 127

        obs_hist = np.array(self.proprio_history_buf, dtype=np.float32).flatten()  # 1270
        self.proprio_history_buf.append(obs_full)

        future_obs = action_mimic.copy()  # 35

        obs_buf = np.concatenate([obs_full, obs_hist, future_obs]).astype(np.float32)  # 1432
        assert obs_buf.shape[0] == TOTAL_OBS_SIZE, (
            f"Expected {TOTAL_OBS_SIZE} obs, got {obs_buf.shape[0]}"
        )

        # -- ONNX inference -----------------------------------------------
        raw_action = self.session.run(
            None, {self.input_name: obs_buf[None]}
        )[0].squeeze().astype(np.float32)

        # Save unclipped action for next obs (matches MuJoCo behaviour)
        self.last_action = raw_action.copy()

        raw_action = np.clip(raw_action, -10.0, 10.0)
        self.pd_target = DEFAULT_DOF_POS + raw_action * ACTION_SCALE

        # -- FPS logging --------------------------------------------------
        now = time.time()
        if self._last_policy_time is not None:
            self._policy_intervals.append(now - self._last_policy_time)
            self._policy_step_count += 1
            if self._policy_step_count % 100 == 0:
                recent = self._policy_intervals[-100:]
                avg_hz = 1.0 / np.mean(recent)
                print(f"[TWIST2] Policy avg Hz (last 100 steps): {avg_hz:.1f}")
        self._last_policy_time = now

    # ------------------------------------------------------------------
    # PD torque servo (500 Hz — every physics step)
    # ------------------------------------------------------------------
    def apply_torques(self):
        joint_positions = self.robot.get_joint_positions()
        joint_velocities = self.robot.get_joint_velocities()

        body_pos = joint_positions[self.body_dof_indices]
        body_vel = joint_velocities[self.body_dof_indices]

        # PD control matching MuJoCo
        torque_body = STIFFNESS * (self.pd_target - body_pos) - DAMPING * body_vel
        torque_body = np.clip(torque_body, -TORQUE_LIMITS, TORQUE_LIMITS)

        full_torque = np.zeros(self._num_dof, dtype=np.float32)
        full_torque[self.body_dof_indices] = torque_body

        # Hand joints: hold at zero with simple PD
        if len(self.hand_dof_indices) > 0:
            hand_pos = joint_positions[self.hand_dof_indices]
            hand_vel = joint_velocities[self.hand_dof_indices]
            hand_torque = 100.0 * (0.0 - hand_pos) - 2.0 * hand_vel
            hand_torque = np.clip(hand_torque, -5.0, 5.0)
            full_torque[self.hand_dof_indices] = hand_torque

        self.robot.set_joint_efforts(full_torque)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def post_reset(self):
        self.robot.post_reset()

        # Reset history buffer
        self.proprio_history_buf.clear()
        for _ in range(HISTORY_LEN):
            self.proprio_history_buf.append(np.zeros(N_OBS_SINGLE, dtype=np.float32))

        self.last_action = np.zeros(NUM_BODY_DOFS, dtype=np.float32)
        self.pd_target = DEFAULT_DOF_POS.copy()
        self.policy_counter = 0
        self._last_policy_time = None
        self._policy_intervals.clear()
        self._policy_step_count = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _find_joint_index(name: str, dof_names: list[str]) -> int | None:
        """Find a joint by name, with fuzzy fallback."""
        # Exact match
        if name in dof_names:
            return dof_names.index(name)
        # Fuzzy: strip _joint suffix and try substring
        stem = name.replace("_joint", "")
        for i, dn in enumerate(dof_names):
            if stem in dn or dn in name:
                return i
        return None
