# AiNex: Sim-to-Real Training & Deployment Plan

## What We Have Today

```
┌─────────────────────────────────────────────────────────────────────┐
│  EXISTING (working, tested)                                        │
│                                                                     │
│  Bridge Server (ws://localhost:9100)                                │
│    ├── MockBackend      (in-memory, no physics)                    │
│    ├── IsaacBackend     (deterministic state surrogate)            │
│    ├── ROS Real Backend (real AiNex hardware)                      │
│    └── ROS Sim Backend  (Gazebo)                                   │
│                                                                     │
│  Protocol: walk.set, walk.command, head.set, action.play,          │
│            policy.start/stop/tick/status, servo.set                 │
│                                                                     │
│  Safety: motion clamping, deadman heartbeat, rate limiting,        │
│          policy heartbeat, manual preemption                        │
│                                                                     │
│  OpenPI Adapter: 11-dim obs → inference → 7-dim action → bridge   │
│  Perception: entity tracking, telemetry fusion, scene summary      │
│  Eliza Plugin: LLM → task instructions → policy lifecycle          │
│  Policy Loops: policy_bridge_loop.py, openpi_loop.py               │
│  Joint Map: 24 joints, servo mapping, pulse↔radian conversion     │
│  Action Library: stand, wave, bow, kick, sit, reset                │
│  Robot Config: URDF, joint limits, actuator params                 │
│  Trace Logging: full command/response/event JSONL audit trail      │
│  149 tests passing                                                  │
└─────────────────────────────────────────────────────────────────────┘
```

## What We're Missing

```
┌─────────────────────────────────────────────────────────────────────┐
│  MISSING (need to build)                                           │
│                                                                     │
│  1. IsaacLab Gym Environment (gymnasium.Env)                       │
│     - No reward functions                                          │
│     - No episode reset logic                                       │
│     - No domain randomization                                      │
│     - No terrain/object spawning                                   │
│     - USD asset not yet generated from URDF                        │
│                                                                     │
│  2. Task-Specific Reward Functions                                 │
│     - Walk-to-target, pick-up-object, visual servoing              │
│                                                                     │
│  3. RL Training Pipeline                                           │
│     - No PPO/SAC/BC training loop                                  │
│     - No policy network architecture                               │
│     - No checkpoint management                                     │
│     - No curriculum learning                                       │
│                                                                     │
│  4. Vision Pipeline                                                │
│     - No camera rendering in sim                                   │
│     - No object detection / segmentation                           │
│     - No depth estimation                                          │
│                                                                     │
│  5. Sim-to-Real Transfer                                           │
│     - No domain randomization config                               │
│     - No real-world evaluation harness                             │
│     - No policy distillation / export                              │
│                                                                     │
│  6. Manipulation                                                   │
│     - Gripper control exists (joints) but no grasp planning        │
│     - No IK solver for reaching                                    │
│     - No contact/force sensing in sim                              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## The Plan: 7 Phases

---

### Phase 1: IsaacLab Gym Environment Foundation

**Goal**: A working `gymnasium.Env` that resets, steps, and renders the AiNex in IsaacLab.

**Why first**: Everything downstream (rewards, training, tasks) requires a proper environment loop.

#### 1a. Generate USD Asset

```bash
# Convert URDF → USD (script exists but USD hasn't been generated)
python3 bridge/isaaclab/convert_urdf_to_usd.py \
  --urdf ros_ws_src/ainex_simulations/ainex_description/urdf/ainex.urdf.xacro \
  --output bridge/generated/ainex.usd
```

Verify with `validate_model.py` — check joint count (24), limits, mesh references.

#### 1b. Create Base Environment

**File**: `training/envs/ainex_env.py`

```
AiNexEnv(gymnasium.Env)
  ├── observation_space: Box(11,) or Dict with image
  ├── action_space: Box(7,) normalized [-1, 1]
  ├── reset() → obs, info
  ├── step(action) → obs, reward, terminated, truncated, info
  └── render() → RGB array (optional)
```

**Observation space** (reuse existing OpenPI adapter normalization):

| Dim | Field | Range (normalized) | Source |
|-----|-------|--------------------|--------|
| 0 | walk_x | [-1, 1] | proprioception |
| 1 | walk_y | [-1, 1] | proprioception |
| 2 | walk_yaw | [-1, 1] | proprioception |
| 3 | walk_height | [-1, 1] | proprioception |
| 4 | walk_speed | [-1, 1] | proprioception |
| 5 | head_pan | [-1, 1] | proprioception |
| 6 | head_tilt | [-1, 1] | proprioception |
| 7 | imu_roll | [-1, 1] | IMU |
| 8 | imu_pitch | [-1, 1] | IMU |
| 9 | is_walking | {-1, 1} | binary |
| 10 | battery | [-1, 1] | battery |

**Extended observation** (for vision tasks):

| Extra Dims | Field | Source |
|------------|-------|--------|
| 11-13 | target_relative_xyz | sim ground truth |
| 14 | target_distance | computed |
| 15 | target_in_view | {0, 1} from camera FOV |
| 64x64x3 | camera_rgb | IsaacLab camera sensor |

**Action space** (maps 1:1 to existing `decode_action` → bridge commands):

| Dim | Field | Denormalized Range |
|-----|-------|--------------------|
| 0 | walk_x | [-0.05, 0.05] m/step |
| 1 | walk_y | [-0.05, 0.05] m/step |
| 2 | walk_yaw | [-10, 10] deg/step |
| 3 | walk_height | [0.015, 0.06] m |
| 4 | walk_speed | [1, 4] integer |
| 5 | head_pan | [-1.5, 1.5] rad |
| 6 | head_tilt | [-1.0, 1.0] rad |

**Reset logic**:
- Teleport robot to spawn position (randomized ±0.5m)
- Set standing pose from `STAND_JOINT_POSITIONS`
- Randomize target object position
- Zero velocities
- Return initial observation

**Step logic**:
- Denormalize action via existing `decode_action()`
- Apply to IsaacLab articulation (joint targets or walk controller)
- Step physics (N substeps at 200Hz, control at 20-50Hz)
- Read back state → build observation via `build_observation()`
- Compute reward (task-specific)
- Check termination (fell over, timeout, success)

#### 1c. Bridge Integration Mode

The env should support two modes:

1. **Direct IsaacLab mode** (fast, for training): env talks directly to Isaac Sim APIs
2. **Bridge mode** (for testing/deployment parity): env routes through bridge websocket

```python
class AiNexEnv:
    def __init__(self, mode="direct", bridge_uri=None):
        if mode == "bridge":
            # Uses existing bridge server + IsaacBackend
            # Same code path as real deployment
            self._backend = BridgeGymWrapper(bridge_uri)
        else:
            # Direct IsaacLab API calls (faster, for mass training)
            self._backend = IsaacLabDirectWrapper(cfg)
```

This ensures **zero sim-to-bridge gap** — policies trained in direct mode can be validated through the bridge before going to real hardware.

#### 1d. Tests

- Env creates, resets, steps without crash
- Observation/action shapes match spec
- Episode terminates on fall (IMU pitch > 45°)
- Bridge mode produces identical trajectories to direct mode

---

### Phase 2: Task-Specific Reward Functions & Environments

**Goal**: Reward functions for 4 concrete tasks, with curriculum difficulty scaling.

#### Task 1: Stable Walking (Foundational)

**File**: `training/envs/tasks/walk_stable.py`

```python
def reward_stable_walk(obs, action, info):
    r = 0.0
    # Reward forward progress
    r += 1.0 * obs.walk_x                          # encourage forward motion
    # Penalize lateral drift
    r -= 0.5 * abs(obs.walk_y)
    # Penalize excessive yaw
    r -= 0.3 * abs(obs.walk_yaw)
    # Reward staying upright
    r += 0.5 * (1.0 - abs(obs.imu_pitch) / 0.5)   # bonus for small pitch
    r += 0.3 * (1.0 - abs(obs.imu_roll) / 0.5)
    # Penalize energy (action magnitude)
    r -= 0.05 * np.sum(action ** 2)
    # Terminal penalty for falling
    if abs(obs.imu_pitch) > 0.8 or abs(obs.imu_roll) > 0.8:
        r -= 10.0  # fell over
    return r
```

**Curriculum**:
- Level 0: Flat ground, no perturbations
- Level 1: Small random force pushes (±2N every 50 steps)
- Level 2: Slight terrain slope (±5°)
- Level 3: Random ground friction (0.5-1.5)

**Success metric**: Walk 2m forward without falling, 10 consecutive episodes.

#### Task 2: Walk to Target (Navigation)

**File**: `training/envs/tasks/walk_to_target.py`

```python
def reward_walk_to_target(obs, action, info):
    r = 0.0
    dist = info["target_distance"]
    prev_dist = info["prev_target_distance"]

    # Shaped reward: getting closer
    r += 5.0 * (prev_dist - dist)

    # Heading reward: face the target
    angle_to_target = info["angle_to_target"]  # radians
    r += 0.5 * (1.0 - abs(angle_to_target) / np.pi)

    # Bonus for reaching target
    if dist < 0.15:  # within 15cm
        r += 50.0

    # Upright bonus
    r += 0.2 * (1.0 - abs(obs.imu_pitch) / 0.5)

    # Fall penalty
    if abs(obs.imu_pitch) > 0.8:
        r -= 10.0

    return r
```

**Episode setup**:
- Target: colored sphere spawned 0.5-3.0m away (curriculum scales distance)
- Robot always starts at origin facing random direction
- Max episode length: 500 steps (at 10Hz = 50 seconds)
- Terminated: reached target (success), fell over (fail), timeout (truncated)

**Curriculum**:
- Level 0: Target 0.5m away, always visible, flat ground
- Level 1: Target 1-2m away, always visible
- Level 2: Target 1-3m away, may be to the side (requires turning)
- Level 3: Target behind robot (requires 180° turn + walk)

#### Task 3: Visual Search & Approach (Target Out of View)

**File**: `training/envs/tasks/visual_search.py`

This extends Task 2 but the target starts **outside the camera's field of view**, requiring the robot to use head pan/tilt to search, then walk toward it.

```python
def reward_visual_search(obs, action, info):
    r = 0.0
    target_visible = info["target_in_camera_fov"]
    dist = info["target_distance"]

    if not target_visible:
        # Reward head scanning behavior
        head_angular_velocity = abs(action[5] - obs.head_pan)  # head_pan change
        r += 0.3 * head_angular_velocity  # encourage looking around
        # Small penalty for walking blindly
        r -= 0.1 * abs(action[0])  # penalize walk_x when can't see target
    else:
        # Target visible — reward centering it in view
        target_pixel_x = info["target_pixel_x"]  # normalized [-1, 1]
        target_pixel_y = info["target_pixel_y"]
        r += 1.0 * (1.0 - abs(target_pixel_x))  # center horizontally
        r += 0.5 * (1.0 - abs(target_pixel_y))  # center vertically
        # Reward approaching
        prev_dist = info["prev_target_distance"]
        r += 5.0 * (prev_dist - dist)
        # Arrival bonus
        if dist < 0.15:
            r += 50.0

    # Upright bonus
    r += 0.2 * (1.0 - abs(obs.imu_pitch) / 0.5)

    return r
```

**Episode setup**:
- Target spawned at random position 1-3m away
- Target initial bearing: 90-270° from robot's forward direction (definitely not in view)
- Camera FOV: ~60° (typical for AiNex onboard camera)
- Robot must: scan → find → turn → approach → reach

**Observation extension for vision tasks**:
```python
# Add to observation dict:
{
    "proprioception": np.array(11,),        # existing 11-dim
    "camera_rgb": np.array(64, 64, 3),      # downsampled camera
    "target_visible": bool,                   # from sim ground truth (for reward only)
}
```

#### Task 4: Pick Up Object (Manipulation)

**File**: `training/envs/tasks/pick_up_object.py`

This is the hardest task. The robot must walk to a ball, reach down, grasp it, and stand back up.

```python
def reward_pick_up(obs, action, info):
    r = 0.0
    phase = info["phase"]  # approach, reach, grasp, lift

    if phase == "approach":
        # Walk toward object (same as Task 2)
        r += 5.0 * (info["prev_dist"] - info["dist"])
        if info["dist"] < 0.3:
            info["phase"] = "reach"
            r += 10.0  # phase transition bonus

    elif phase == "reach":
        # Reward lowering body + extending arm toward object
        gripper_to_object = info["gripper_to_object_dist"]
        prev_gripper_dist = info["prev_gripper_to_object_dist"]
        r += 5.0 * (prev_gripper_dist - gripper_to_object)
        if gripper_to_object < 0.05:
            info["phase"] = "grasp"
            r += 10.0

    elif phase == "grasp":
        # Reward closing gripper
        r += 10.0 * info["gripper_contact_force"]  # from contact sensor
        if info["object_grasped"]:
            info["phase"] = "lift"
            r += 20.0

    elif phase == "lift":
        # Reward standing back up with object
        object_height = info["object_height"]
        r += 10.0 * object_height  # higher = better
        if object_height > 0.15 and info["object_grasped"]:
            r += 100.0  # success!

    # Always: upright bonus, fall penalty
    r += 0.2 * (1.0 - abs(obs.imu_pitch) / 0.5)
    if abs(obs.imu_pitch) > 0.8:
        r -= 10.0

    return r
```

**Extended action space for manipulation**:
The existing 7-dim action space controls walk + head. For manipulation, we need arm control:

| Dim | Field | Range |
|-----|-------|-------|
| 0-6 | (existing walk + head) | same |
| 7 | r_sho_pitch | [-2.09, 2.09] rad |
| 8 | r_sho_roll | [-2.09, 2.09] rad |
| 9 | r_el_pitch | [-2.09, 2.09] rad |
| 10 | r_el_yaw | [-2.09, 2.09] rad |
| 11 | r_gripper | [-2.09, 2.09] rad (open/close) |
| 12-16 | left arm (same layout) | same |

Total: **17-dim action space** for full-body tasks, or **12-dim** for single-arm tasks.

**This is hard.** Recommended approach:
1. Pre-train walk policy (Task 1-2) and freeze leg control
2. Train arm-only policy on top with walk policy providing locomotion
3. Combined fine-tuning

**Episode setup**:
- Ball (5cm diameter) spawned 0.5-2m away on ground
- Robot starts standing
- Max 1000 steps at 10Hz (100 seconds)
- Success: ball lifted 15cm+ off ground while grasped

---

### Phase 3: RL Training Pipeline

**Goal**: Train policies on the tasks above using PPO, with checkpoint management and evaluation.

#### 3a. Policy Network Architecture

**File**: `training/models/policy_net.py`

```python
class AiNexPolicy(nn.Module):
    """MLP policy for proprioceptive control, CNN+MLP for vision tasks."""

    def __init__(self, obs_dim=11, act_dim=7, hidden=256, use_vision=False):
        # Proprioception encoder
        self.proprio_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.ELU(),
        )
        # Optional vision encoder
        if use_vision:
            self.vision_encoder = nn.Sequential(
                nn.Conv2d(3, 32, 8, stride=4), nn.ELU(),
                nn.Conv2d(32, 64, 4, stride=2), nn.ELU(),
                nn.Flatten(),
                nn.Linear(64 * 6 * 6, hidden), nn.ELU(),
            )
        # Actor head (Gaussian)
        self.actor_mean = nn.Linear(hidden, act_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(act_dim))
        # Critic head
        self.critic = nn.Linear(hidden, 1)
```

**Why MLP first**: The 11-dim proprioceptive observation is small. MLP trains fast (minutes on GPU), and the existing OpenPI adapter already handles normalization. Vision can be bolted on later.

#### 3b. Training Script

**File**: `training/train_ppo.py`

```
Usage:
  python3 -m training.train_ppo \
    --task walk_stable \
    --num-envs 4096 \
    --total-steps 50_000_000 \
    --checkpoint-dir checkpoints/walk_stable/ \
    --wandb-project ainex
```

Key parameters:
- PPO with GAE (λ=0.95, γ=0.99)
- Clipping: ε=0.2
- Learning rate: 3e-4 with linear decay
- Batch size: 4096 envs × 32 steps = 131k transitions per update
- Entropy coefficient: 0.01 (encourage exploration)
- 4096 parallel environments in IsaacLab (GPU-accelerated)

#### 3c. Evaluation & Checkpointing

**File**: `training/eval/evaluate_policy.py`

```
Usage:
  python3 -m training.eval.evaluate_policy \
    --checkpoint checkpoints/walk_stable/best.pt \
    --task walk_stable \
    --num-episodes 100 \
    --render
```

Metrics logged per evaluation:
- Success rate (task-specific)
- Episode length (mean, std)
- Cumulative reward (mean, std)
- Stability (max IMU deviation)
- Action smoothness (consecutive action L2 distance)

Checkpointing:
- Save every 1M steps
- Keep best by evaluation success rate
- Export to ONNX for deployment

---

### Phase 4: Vision Pipeline

**Goal**: Camera rendering in sim, object detection, and visual observation encoding.

#### 4a. Camera Sensor in IsaacLab

Add camera to the robot's head link:
```python
# In env setup
camera_cfg = CameraCfg(
    prim_path="/World/AiNex/head_tilt_link/camera",
    update_period=0.1,  # 10Hz matching policy tick rate
    height=64, width=64,
    data_types=["rgb", "distance_to_camera"],
    spawn=PinholeCameraCfg(
        focal_length=2.0,
        horizontal_aperture=3.6,  # ~60° FOV
    ),
)
```

#### 4b. Object Spawning

```python
# Spawn task objects with domain randomization
ball_cfg = RigidObjectCfg(
    prim_path="/World/ball",
    spawn=SphereCfg(radius=0.025, mass=0.05,  # 5cm ball, 50g
        visual_material=PreviewSurfaceCfg(diffuse_color=random_color()),
    ),
)
```

Domain randomization per episode:
- Ball color (red, blue, green, yellow, random)
- Ball size (3-7cm diameter)
- Lighting direction and intensity
- Ground texture
- Camera noise (Gaussian, ±5 pixel jitter)

#### 4c. Visual Encoder Training

Two options:

**Option A: End-to-end** — CNN directly in the policy network (simpler, works for sim)

**Option B: Pre-trained encoder** — Use a frozen vision backbone (e.g., DINOv2-small) and train a lightweight head on top (better for sim-to-real transfer since the backbone generalizes across visual domains)

Recommended: **Option B** for any task requiring sim-to-real transfer.

```python
class VisionPolicy(nn.Module):
    def __init__(self):
        self.vision_backbone = load_dinov2_small()  # frozen
        self.vision_head = nn.Linear(384, 128)
        self.proprio_encoder = nn.Linear(11, 128)
        self.actor = nn.Linear(256, 7)
        self.critic = nn.Linear(256, 1)

    def forward(self, proprio, image):
        v = self.vision_head(self.vision_backbone(image))
        p = self.proprio_encoder(proprio)
        x = torch.cat([v, p], dim=-1)
        return self.actor(x), self.critic(x)
```

#### 4d. Perception Bridge

Connect sim camera → PerceptionAggregator:
- Run object detector on camera frames in sim (ground truth bounding boxes available)
- Feed detected entities into `PerceptionAggregator.update_entities_batch()`
- Snapshot feeds into `build_observation()` for the OpenPI adapter

This keeps the **exact same perception pipeline** in sim and on real hardware.

---

### Phase 5: OpenPI Integration for Language-Conditioned Policies

**Goal**: Train a single policy that responds to natural language task instructions.

#### 5a. Why OpenPI

OpenPI provides a standard interface for **language-conditioned visuomotor policies**. Instead of training separate policies for each task, one network handles multiple tasks conditioned on the prompt:

```
"walk forward" → steady forward gait
"walk to the red ball" → visual search + navigation
"pick up the object" → approach + grasp + lift
"wave hello" → trigger wave action
```

#### 5b. Multi-Task Training

```python
class MultiTaskEnv(AiNexEnv):
    TASKS = {
        "walk_forward": (walk_stable_reward, walk_stable_reset),
        "walk_to_target": (walk_to_target_reward, target_reset),
        "find_and_approach": (visual_search_reward, search_reset),
        "pick_up_ball": (pick_up_reward, manipulation_reset),
    }

    def reset(self):
        # Randomly sample a task
        self.current_task = random.choice(list(self.TASKS.keys()))
        self.reward_fn, reset_fn = self.TASKS[self.current_task]
        obs = reset_fn(self)
        obs["prompt"] = self.current_task
        return obs
```

#### 5c. Training → OpenPI Server

1. Train multi-task policy with language conditioning
2. Export model weights
3. Wrap in OpenPI-compatible inference server:

```python
# training/serve_policy.py
from fastapi import FastAPI
app = FastAPI()

policy = load_checkpoint("checkpoints/multitask/best.pt")

@app.post("/infer")
async def infer(obs: dict):
    action = policy(obs["state"], obs["prompt"], obs.get("image"))
    return {"action": action.tolist(), "confidence": float(confidence)}
```

4. Point existing `openpi_loop.py` at this server:

```bash
python3 -m training.runtime.openpi_loop \
  --bridge-uri ws://localhost:9100 \
  --openpi-url http://localhost:8000 \
  --task "walk to the red ball" \
  --hz 10 --max-steps 500
```

**This requires zero changes to bridge, safety, or deployment code.** The OpenPI adapter, bridge server, and safety clamping all work unchanged.

---

### Phase 6: Sim-to-Real Transfer

**Goal**: Policies trained in IsaacLab work on the real AiNex robot.

#### 6a. Domain Randomization Checklist

| Parameter | Sim Range | Why |
|-----------|-----------|-----|
| Ground friction | 0.3 - 1.5 | Real floors vary |
| Joint stiffness | ×0.8 - ×1.2 | Servo behavior varies |
| Joint damping | ×0.5 - ×2.0 | Wear and temperature |
| IMU noise | ±0.05 rad | Sensor noise |
| Battery voltage | 10.4 - 12.6 V | Real battery drains |
| Action delay | 0 - 50ms | Communication latency |
| Observation noise | ±2% Gaussian | Sensor quantization |
| Mass randomization | ×0.9 - ×1.1 | Manufacturing tolerance |
| Camera color jitter | ±10% HSV | Lighting conditions |
| External force push | 0 - 3N random | Human/wind perturbation |

#### 6b. Transfer Pipeline

```
1. Train in IsaacLab (Phase 3)     — 4096 parallel envs, millions of steps
                ↓
2. Evaluate in IsaacLab            — 100+ episodes, success rate > 90%
                ↓
3. Test via bridge + IsaacBackend  — verify bridge protocol path works
                ↓
4. Test via bridge + Gazebo        — ROS sim with different physics
                ↓
5. Deploy via bridge + ROS Real    — real robot, tethered, safety limits tight
                ↓
6. Relax safety limits gradually   — increase speed/stride as confidence grows
```

**The bridge abstraction is the key enabler here.** Steps 3-6 use the exact same policy code (`openpi_loop.py`), just pointing at different bridge backends:

```bash
# Step 3: Isaac backend
python3 -m bridge.server --backend isaac --port 9100

# Step 4: Gazebo
python3 -m bridge.server --backend ros_sim --port 9100

# Step 5: Real robot
python3 -m bridge.server --backend ros_real --port 9100
```

#### 6c. Real-World Safety Protocol

When deploying to real hardware:

1. **Tether the robot** — prevent falls from damaging hardware
2. **Start with walk_speed=1** — override policy speed to minimum
3. **Reduce motion bounds** — halve POLICY_WALK_X_MAX etc. initially
4. **Set max_steps=50** — short episodes while evaluating
5. **Monitor IMU** — auto-stop if pitch/roll exceed 30° (not 45°)
6. **Record traces** — every command/response logged for post-mortem
7. **Human kill switch** — Eliza `ROBOT_IDLE` or ctrl-C on policy loop

Gradual relaxation:
```
Week 1: speed=1, stride=0.02, 50 steps  → stable walking?
Week 2: speed=2, stride=0.03, 100 steps → navigation?
Week 3: speed=2, stride=0.05, 200 steps → full task execution?
```

---

### Phase 7: Eliza Orchestration (High-Level Task Execution)

**Goal**: Natural language → multi-step autonomous behavior.

#### 7a. Architecture

```
User: "Pick up the red ball near the table"
         ↓
    Eliza Agent (GPT-4o)
         ↓ decomposes into steps
    Step 1: POLICY_START(task="find the red ball", max_steps=200)
         ↓ policy runs, finds ball, approaches
    Step 2: POLICY_START(task="pick up the ball", max_steps=300)
         ↓ policy runs manipulation sequence
    Step 3: "I've picked up the red ball"
         ↓
User sees response + can give next instruction
```

#### 7b. Task Decomposition via Providers

The Eliza plugin already has providers that give the LLM real-time state:

```
ROBOT_STATE → battery, walking, IMU, walk params, head position
PERCEPTION_STATE → tracked entities with positions and confidence
POLICY_STATE → active/idle, current task, step count
```

The LLM uses this context to decide what to do next:

```
"I can see a red ball at position (0.3, -0.2, 1.5m). It's 1.5m away
and slightly to my left. I'll start a walk-to-target policy."
→ POLICY_START(task="walk to the red ball", hz=10, max_steps=200)
```

#### 7c. Failure Recovery

```
Policy fails (fell, timeout, low confidence)
         ↓
    POLICY_STATE provider reports: idle, reason="low_confidence"
         ↓
    Eliza Agent: "My confidence dropped. Let me look around first."
         ↓
    HEAD_SET(pan=0.5, tilt=-0.2) — look to the left
         ↓
    PERCEPTION_STATE updates with new entity sightings
         ↓
    POLICY_START(task="approach detected object", max_steps=100)
```

---

## Implementation Priority & Dependencies

```
Phase 1 ──→ Phase 2 ──→ Phase 3 ──→ Phase 6
  (env)      (rewards)   (training)   (sim2real)
                            │
                            ↓
Phase 4 ─────────────→ Phase 5 ──→ Phase 7
  (vision)              (openpi)    (eliza)
```

**Recommended order**:
1. **Phase 1** (1-2 weeks): Get IsaacLab env working
2. **Phase 2, Task 1-2** (1 week): Walk + navigate rewards
3. **Phase 3** (1-2 weeks): PPO training pipeline, train walking
4. **Phase 6a** (1 week): Domain randomization
5. **Phase 4** (1-2 weeks): Camera + vision (parallel with Phase 6)
6. **Phase 2, Task 3-4** (1-2 weeks): Visual search + manipulation rewards
7. **Phase 5** (1 week): OpenPI server wrapping
8. **Phase 6b-c** (ongoing): Sim-to-real transfer
9. **Phase 7** (1 week): Eliza orchestration refinement

## Concrete First Steps

If starting today:

```bash
# 1. Generate USD asset
python3 bridge/isaaclab/convert_urdf_to_usd.py

# 2. Create training/envs/ directory structure
mkdir -p training/envs/tasks

# 3. Write AiNexEnv with MockBackend first (no IsaacLab needed)
#    - Uses existing SimRobotState for basic physics
#    - Tests env API contract (reset, step, obs/act shapes)

# 4. Wire up IsaacLab when USD asset is ready
#    - run_sim.py already handles spawning
#    - Add gymnasium wrapper around the sim loop

# 5. Implement walk_stable reward
#    - Simplest task, validates entire pipeline

# 6. Train first policy with PPO
#    - Even with MockBackend physics, validates training loop
#    - Switch to IsaacLab for real physics training
```

## What We Don't Need to Build

Thanks to the existing infrastructure:

- **Protocol handling** — bridge server already handles everything
- **Safety** — motion clamping, deadman, heartbeats all working
- **Deployment path** — openpi_loop.py + bridge + any backend
- **Observation/action encoding** — OpenPI adapter handles normalization
- **Perception** — PerceptionAggregator already tracks entities
- **High-level orchestration** — Eliza plugin already wired up
- **Trace logging** — full JSONL audit trail for debugging
- **Testing** — 149 tests validating the entire stack

The gap is specifically: **gym environment + reward functions + training loop + vision**. Everything else is built and tested.
