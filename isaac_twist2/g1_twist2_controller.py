import json
import time
from collections import deque

import numpy as np
import onnxruntime as ort
import redis
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.prims import define_prim, get_prim_at_path
from isaacsim.core.utils.rotations import quat_to_rot_matrix
from isaacsim.core.utils.types import ArticulationAction

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

# Inspire hand: 6 primary joints per hand (controllable via drive)
INSPIRE_HAND_JOINT_NAMES_LEFT = [
    "L_thumb_proximal_yaw_joint", "L_thumb_proximal_pitch_joint",
    "L_index_proximal_joint", "L_middle_proximal_joint",
    "L_ring_proximal_joint", "L_pinky_proximal_joint",
]
INSPIRE_HAND_JOINT_NAMES_RIGHT = [
    "R_thumb_proximal_yaw_joint", "R_thumb_proximal_pitch_joint",
    "R_index_proximal_joint", "R_middle_proximal_joint",
    "R_ring_proximal_joint", "R_pinky_proximal_joint",
]

# Standard G1 Dex3 hand: 7 joints per hand
# Left: thumb(0,1,2), middle(0,1), index(0,1)
DEX3_HAND_JOINT_NAMES_LEFT = [
    "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint", "left_hand_middle_1_joint",
    "left_hand_index_0_joint", "left_hand_index_1_joint",
]
# Right: thumb(0,1,2), index(0,1), middle(0,1)
DEX3_HAND_JOINT_NAMES_RIGHT = [
    "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
    "right_hand_index_0_joint", "right_hand_index_1_joint",
    "right_hand_middle_0_joint", "right_hand_middle_1_joint",
]

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
        control_mode: str = "pd",
    ):
        # -- Validate control mode ----------------------------------------
        if control_mode not in ("torque", "pd"):
            raise ValueError(f"control_mode must be 'torque' or 'pd', got '{control_mode}'")
        self.control_mode = control_mode

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

        # Hand control state (filled during initialize)
        self.hand_left_indices: np.ndarray | None = None
        self.hand_right_indices: np.ndarray | None = None
        self.hand_mimic_indices: np.ndarray | None = None
        self.hand_target_left: np.ndarray | None = None
        self.hand_target_right: np.ndarray | None = None
        self.num_hand_dofs_per_side: int = 0
        self.hand_kp: np.ndarray | None = None
        self.hand_kd: np.ndarray | None = None
        self.hand_max_effort: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def initialize(self, physics_sim_view=None):
        self.robot.initialize(physics_sim_view=physics_sim_view)
        self._num_dof = self.robot.num_dof

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

        # -- Detect hand variant and build hand indices ------------------
        hand_dof_names = [dof_names[i] for i in self.hand_dof_indices]
        is_inspire = any(n.startswith("L_") or n.startswith("R_") for n in hand_dof_names)

        if is_inspire:
            left_names = INSPIRE_HAND_JOINT_NAMES_LEFT
            right_names = INSPIRE_HAND_JOINT_NAMES_RIGHT
        else:
            left_names = DEX3_HAND_JOINT_NAMES_LEFT
            right_names = DEX3_HAND_JOINT_NAMES_RIGHT

        self.hand_left_indices = np.array(
            [self._find_joint_index(n, dof_names) for n in left_names], dtype=np.int64
        )
        self.hand_right_indices = np.array(
            [self._find_joint_index(n, dof_names) for n in right_names], dtype=np.int64
        )
        self.num_hand_dofs_per_side = len(left_names)

        # Mimic indices: hand DOFs that are not primary controllable
        primary_set = set(self.hand_left_indices) | set(self.hand_right_indices)
        self.hand_mimic_indices = np.array(
            sorted(set(self.hand_dof_indices) - primary_set), dtype=np.int64
        )

        self.hand_target_left = np.zeros(self.num_hand_dofs_per_side, dtype=np.float32)
        self.hand_target_right = np.zeros(self.num_hand_dofs_per_side, dtype=np.float32)

        hand_variant = "Inspire" if is_inspire else "Dex3"
        print(f"[TWIST2] Hand variant: {hand_variant}")
        print(f"[TWIST2]   Left primary ({len(self.hand_left_indices)}): {self.hand_left_indices}")
        print(f"[TWIST2]   Right primary ({len(self.hand_right_indices)}): {self.hand_right_indices}")
        print(f"[TWIST2]   Mimic ({len(self.hand_mimic_indices)}): {self.hand_mimic_indices}")

        # -- Build full-articulation default arrays ----------------------
        self.full_default_pos = np.zeros(self._num_dof, dtype=np.float32)
        self.full_default_pos[self.body_dof_indices] = DEFAULT_DOF_POS

        # -- Read existing USD gains before overriding -------------------
        existing_kp, existing_kd = self.robot._articulation_view.get_gains()
        existing_max = self.robot._articulation_view.get_max_efforts()
        hand_ctrl_indices = np.concatenate([self.hand_left_indices, self.hand_right_indices])

        # -- Control-mode setup ------------------------------------------
        if self.control_mode == "torque":
            # Explicit torque control: we compute PD torques ourselves
            self.robot.get_articulation_controller().set_effort_modes("force")
            self.robot.get_articulation_controller().switch_control_mode("effort")

            # Store hand PD gains from USD for manual torque computation
            self.hand_kp = existing_kp[0][hand_ctrl_indices].copy()
            self.hand_kd = existing_kd[0][hand_ctrl_indices].copy()
            self.hand_max_effort = existing_max[0][hand_ctrl_indices].copy()

            zero_gains = np.zeros(self._num_dof, dtype=np.float32)
            self.robot._articulation_view.set_gains(zero_gains, zero_gains)

            high_effort = np.full(self._num_dof, 1000.0, dtype=np.float32)
            self.robot._articulation_view.set_max_efforts(high_effort)
        else:
            # PhysX implicit PD: set gains and send position targets
            self.robot.get_articulation_controller().switch_control_mode("position")

            full_kp = np.zeros(self._num_dof, dtype=np.float32)
            full_kd = np.zeros(self._num_dof, dtype=np.float32)
            full_kp[self.body_dof_indices] = STIFFNESS
            full_kd[self.body_dof_indices] = DAMPING
            # Hand controllable joints: use USD values
            if len(hand_ctrl_indices) > 0:
                full_kp[hand_ctrl_indices] = existing_kp[0][hand_ctrl_indices]
                full_kd[hand_ctrl_indices] = existing_kd[0][hand_ctrl_indices]
            # Mimic indices stay at zero — PhysX mimic constraint drives them
            self.robot._articulation_view.set_gains(full_kp, full_kd)

            full_max = np.zeros(self._num_dof, dtype=np.float32)
            full_max[self.body_dof_indices] = TORQUE_LIMITS
            if len(hand_ctrl_indices) > 0:
                full_max[hand_ctrl_indices] = existing_max[0][hand_ctrl_indices]
            self.robot._articulation_view.set_max_efforts(full_max)

        print(f"[TWIST2] Control mode: {self.control_mode}")

        # Publish initial state to Redis
        self.redis_pipeline.set(_REDIS_STATE_BODY, json.dumps(np.zeros(34).tolist()))
        self.redis_pipeline.set(_REDIS_STATE_HAND_L,
            json.dumps(np.zeros(self.num_hand_dofs_per_side).tolist()))
        self.redis_pipeline.set(_REDIS_STATE_HAND_R,
            json.dumps(np.zeros(self.num_hand_dofs_per_side).tolist()))
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
        self.redis_pipeline.set(_REDIS_STATE_HAND_L,
            json.dumps(joint_positions[self.hand_left_indices].tolist()))
        self.redis_pipeline.set(_REDIS_STATE_HAND_R,
            json.dumps(joint_positions[self.hand_right_indices].tolist()))
        self.redis_pipeline.set(_REDIS_STATE_NECK, json.dumps(np.zeros(2).tolist()))
        self.redis_pipeline.set(_REDIS_T_STATE, int(time.time() * 1000))
        self.redis_pipeline.execute()

        # -- Read mimic actions from Redis --------------------------------
        for key in [_REDIS_ACTION_BODY, _REDIS_ACTION_HAND_L, _REDIS_ACTION_HAND_R, _REDIS_ACTION_NECK]:
            self.redis_pipeline.get(key)
        redis_results = self.redis_pipeline.execute()

        action_mimic = np.array(json.loads(redis_results[0]), dtype=np.float32) if redis_results[0] else np.zeros(N_MIMIC_OBS, dtype=np.float32)

        # Parse hand actions from Redis
        action_hand_left_raw = redis_results[1]
        action_hand_right_raw = redis_results[2]

        if action_hand_left_raw:
            parsed = np.array(json.loads(action_hand_left_raw), dtype=np.float32)
            if len(parsed) == self.num_hand_dofs_per_side:
                self.hand_target_left = parsed

        if action_hand_right_raw:
            parsed = np.array(json.loads(action_hand_right_raw), dtype=np.float32)
            if len(parsed) == self.num_hand_dofs_per_side:
                self.hand_target_right = parsed

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

        # Hand controllable joints: PD with targets from Redis
        if len(self.hand_left_indices) > 0:
            hand_ctrl = np.concatenate([self.hand_left_indices, self.hand_right_indices])
            hand_target = np.concatenate([self.hand_target_left, self.hand_target_right])
            hand_pos = joint_positions[hand_ctrl]
            hand_vel = joint_velocities[hand_ctrl]
            hand_torque = self.hand_kp * (hand_target - hand_pos) - self.hand_kd * hand_vel
            hand_torque = np.clip(hand_torque, -self.hand_max_effort, self.hand_max_effort)
            full_torque[hand_ctrl] = hand_torque
        # Mimic joint torques stay at zero

        self.robot.set_joint_efforts(full_torque)

    # ------------------------------------------------------------------
    # PhysX implicit PD targets (100 Hz — every policy step)
    # ------------------------------------------------------------------
    def apply_pd_targets(self):
        """Send position targets to PhysX implicit PD solver."""
        full_target = self.full_default_pos.copy()
        full_target[self.body_dof_indices] = self.pd_target
        full_target[self.hand_left_indices] = self.hand_target_left
        full_target[self.hand_right_indices] = self.hand_target_right
        # Mimic indices stay at default (zero) — PhysX mimic constraint drives them
        self.robot.get_articulation_controller().apply_action(
            ArticulationAction(joint_positions=full_target)
        )

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

        if self.hand_target_left is not None:
            self.hand_target_left[:] = 0.0
        if self.hand_target_right is not None:
            self.hand_target_right[:] = 0.0

        if self.control_mode == "pd":
            self.apply_pd_targets()

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
