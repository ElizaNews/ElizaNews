# AiNex End-to-End Roadmap

This roadmap turns the current ROS-based AiNex stack into a unified sim/real ML platform.

## Phase 0: Freeze Interface Contract [DONE]

Target outcome:

- one websocket protocol used by all clients and all robot backends
- versioned command/event envelopes
- replayable command traces for parity tests

Status: **Complete.**

- `bridge/protocol.py` — CommandEnvelope, ResponseEnvelope, EventEnvelope with strict schema
- `bridge/validation.py` — per-command payload validation (walk.set, head.set, action.play, servo.set, policy.*)
- `bridge/trace_log.py` — thread-safe JSONL append logger for all commands/responses
- `bridge/tools/replay_trace.py` — replay recorded traces against any backend
- `bridge/examples/basic_walk_trace.jsonl` — example trace
- Protocol version 1.0.0 declared in `bridge/config/default_bridge_config.json`

Remaining gap: protocol version field not enforced in envelopes at runtime (config declares it, but no negotiation or validation on connect).

## Phase 1: Real Robot Backend Hardening [DONE]

Tasks:

1. Rate limiting and deadman stop. **Done.**
   - `bridge/safety.py` — `CommandRateLimiter` (30 Hz sliding window), deadman heartbeat detection (1s timeout), auto walk.stop with preempt flag
2. Action queue and cancellation policy. **Done.**
   - `bridge/server.py` — bounded asyncio.Queue (256), preempt flag clears queue, backpressure on full queue
3. Servo safety limits on raw command path. **Partial.**
   - Policy motion bounds enforced in `bridge/safety.py` (walk x/y/yaw/height/speed, head pan/tilt)
   - Joint limits defined in `bridge/isaaclab/ainex_cfg.py` (`build_joint_limits()`)
   - Gap: `servo.set` command does not validate against joint limits before forwarding
4. Full telemetry stream. **Partial.**
   - Battery: done (ROS `/ros_robot_controller/battery`, published in `telemetry.basic` at 2 Hz)
   - IMU: roll/pitch only (no yaw/heading, no raw accel/gyro)
   - Walk state: done (is_walking, walk params)
   - Camera metadata: not implemented (Gazebo plugin exists but bridge doesn't subscribe)
5. Bridge startup systemd unit. **Done.**
   - `bridge/systemd/ainex-bridge.service` — auto-restart, configurable args

Acceptance checks:

- upstream agent controls robot only through bridge: **yes**
- all commands and effects are structured-logged: **yes** (trace_log.py)

## Phase 2: Sim Parity Backend [PARTIAL]

Tasks:

1. Run existing Gazebo stack with `ainex_controller` sim mode. **Exists but not bridged.**
   - Gazebo URDF/xacro, launch files, worlds all exist in `ros_ws_src/ainex_simulations/`
   - Gazebo plugins configured (IMU with noise, camera 640x480 @ 30 Hz)
   - No integration layer that runs Gazebo and feeds telemetry back through bridge
2. Bind `ros_sim` backend to the same command API. **Skeleton only.**
   - `bridge/backends/ros_backend.py` connects to ROS topics but has no sim runner
   - `bridge_targets.json` lists a `sim` target but it points at the same ROS backend
3. Add playback harness for identical traces. **Done.**
   - `bridge/tools/parity_check.py` — runs same trace against two URIs, compares responses
   - `bridge/tests/test_backend_parity.py` — Isaac vs Mock parity assertions
4. Measure drift in command result, latency, and walk-state transitions. **Not done.**
   - No parallel trajectory comparison or divergence measurement

## Phase 3: Isaac Integration [MOSTLY DONE]

Tasks:

1. Build Isaac articulation model from AiNex URDF assets. **Done.**
   - `bridge/isaaclab/ainex_cfg.py` — full articulation config, 24 joints, actuator params
   - `bridge/isaaclab/joint_map.py` — LEG/ARM/HEAD joint name maps
   - `bridge/isaaclab/convert_urdf_to_usd.py` — URDF-to-USD converter
   - `bridge/isaaclab/run_sim.py` — simulation runner with AppLauncher
2. Mirror bridge commands and telemetry in Isaac adapter. **Done.**
   - `bridge/backends/isaac_backend.py` — command-envelope protocol
   - `bridge/backends/rosbridge_isaac.py` — ROSBridge-compatible protocol
   - `bridge/isaaclab/sim_state.py` — SimRobotState with walk/head/servo/battery/IMU
3. Add camera and IMU sensor simulation. **Partial.**
   - IMU: basic quaternion orientation in SimRobotState (roll/pitch when walking)
   - Camera: not implemented (capability declared but no sensor pipeline)
4. Add domain randomization. **Done (in MuJoCo, not Isaac).**
   - `training/mujoco/domain_randomization.py` — CPU-based: friction, mass, stiffness, damping (5 presets)
   - `training/mujoco/randomize.py` — JAX/MJX vmap: friction, frictionloss, armature, mass, COM, qpos0, gains, damping
   - Gap: no DR wired into IsaacLab envs (only MuJoCo training uses it)

Acceptance checks:

- existing bridge client controls Isaac without API changes: **yes** (isaac_backend.py, tested in test_backend_parity.py)

## Phase 4: Policy Training Stack [IN PROGRESS - TRAINING NOW]

Tasks:

1. Create offline dataset format for trajectories and action labels. **Not done.**
   - `eval_policy.py` collects in-memory trajectories but doesn't persist them
   - `interfaces.py` has `PolicyTransitionRecord` for trace logging but not offline RL datasets
2. Implement behavior cloning baseline for movement primitives. **Not done.**
3. Implement RL fine-tuning curriculum for locomotion objectives. **Core done, curriculum not automated.**
   - `training/mujoco/train.py` — Brax PPO with 4096 parallel MJX envs, GPU-accelerated
   - `training/mujoco/joystick.py` — velocity-tracking locomotion env (14 reward terms)
   - `training/mujoco/target.py` — target-reaching env with distance shaping
   - Domain randomization integrated into training loop
   - 50M-step training run completed with checkpoints
   - Gap: no curriculum scheduler (manual config changes only, no auto-progression)
4. Add policy-vector conditioning (`z`) from agent embedding pipeline. **Designed, not integrated.**
   - `training/rl/meta/text_encoder.py` — SentenceTransformer (384-dim) + BoW fallback
   - `training/rl/meta/meta_policy.py` — text embedding + robot state -> skill logits + params
   - Gap: meta-policy not integrated into MuJoCo training loop
5. Add fallback hand-authored controller gating for unsafe policy states. **Partial.**
   - Bridge safety layer (clamping, deadman, heartbeat) provides hardcoded gating
   - Skill registry with stand/walk/turn/wave/bow fallbacks
   - Gap: no learned unsafe-state detector or adaptive gating

Acceptance checks:

- model executes `walk`, `turn`, `wave` and objective tasks with bounded fall rate: **locomotion training in progress; skills exist as hand-authored fallbacks**

## Phase 4.5: Eliza + OpenPI Live Policy [DONE]

Target outcome:

- Eliza agent runs high-level dialog/planning through ElizaOS runtime
- Robot controlled via bridge websocket with full safety gating
- Live OpenPI policy mode streams observations and applies actions
- Seamless switching between manual control and policy mode

Components delivered:

1. **Bridge protocol extensions** — `policy.start/stop/tick/status` commands, `telemetry.policy`, `safety.policy_guard` events. `bridge/protocol.py`
2. **Policy safety layer** — motion bounds clamping, policy heartbeat monitoring (2s timeout), auto-fallback on stale ticks. `bridge/safety.py`
3. **OpenPI adapter** — 11-dim normalized observation builder, action decoder with clamping, bridge command generation. `bridge/openpi_adapter.py`
4. **Perception aggregator** — entity tracking (64 max, 5s stale timeout), telemetry fusion, scene summaries. `bridge/perception.py`
5. **Eliza Python plugin** (`eliza/elizaos_plugin_ainex/`) — 7 actions (walk, head, action_play, policy start/stop, idle) + 3 providers (robot state, perception, policy state). 1,078 LOC across 5 files.
6. **Integration tests** — 149 tests across 21 files (exceeds 45+ target). `bridge/tests/`

Status: **Complete.** All acceptance checks pass.

## Phase 5: Perception + Egocentric Navigation [NOT STARTED]

Tasks:

1. Add monocular odometry/SLAM module.
2. Integrate pose uncertainty into policy gating.
3. Train camera-conditioned locomotion policy variants.
4. Add vision failure modes to robustness tests.

## Phase 6: Sim2Real Rollout [NOT STARTED]

Tasks:

1. Add regression suite for safety and behavior parity.
2. Tethered deployment with watchdog stop.
3. Closed-area deployment with telemetry logging.
4. Incremental production rollout.

---

## Current Focus

**Phase 4 — RL locomotion training is running.** MuJoCo Playground pipeline (`training/mujoco/`) is the single training path. Brax PPO on MJX with 4096 parallel envs, domain randomization, joystick velocity-tracking and target-reaching envs. A 50M-step run has completed; further training in progress.

Key files:
- `training/mujoco/train.py` — training entry point
- `training/mujoco/joystick.py` — locomotion env
- `training/mujoco/target.py` — target-reaching env
- `training/mujoco/inference.py` — checkpoint loading + inference
- `training/mujoco/eval_policy.py` — rollout + GIF rendering
- `training/mujoco/domain_randomization.py` — physics randomization
- `checkpoints/mujoco_locomotion/` — trained params + config
