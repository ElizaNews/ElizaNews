# OpenPI + Eliza AiNex Integration Guide

## Architecture

```
ElizaAgent (LLM) ──> AiNex Plugin ──> BridgeClient (ws) ──> BridgeServer ──> Backend (mock/ros/isaac)
                                                                ↑
PerceptionAggregator ──────────────────────────────────────────┘
                                                                ↑
OpenPIAdapter ──> OpenPI Client (inference) ──> policy.tick ───┘
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AINEX_BRIDGE_URL` | `ws://localhost:9100` | Bridge websocket URL |
| `AINEX_BRIDGE_BACKEND` | `mock` | Backend: mock, ros_real, ros_sim, isaac |
| `OPENAI_API_KEY` | (required) | OpenAI API key for Eliza LLM |
| `OPENAI_LARGE_MODEL` | `gpt-4o` | Model for Eliza agent reasoning |
| `OPENPI_SERVER_URL` | (optional) | OpenPI inference server URL |

## Startup Order

1. **Start the robot / simulator**
   ```bash
   # Real robot: ensure ROS master is running
   roslaunch ainex_controller ainex_control.launch

   # Sim: launch Gazebo
   roslaunch ainex_gazebo ainex_gazebo.launch
   ```

2. **Start the bridge server**
   ```bash
   cd /home/shaw/Documents/ainex-robot-code
   python3 -m bridge.server --backend ros_real --port 9100 \
     --deadman-timeout-sec 2.0 --trace-log-path traces/session.jsonl
   ```

3. **Start the Eliza agent** (optional, for LLM-driven control)
   ```python
   from elizaos_plugin_ainex.agent import AiNexRobotAgent

   agent = AiNexRobotAgent(bridge_url="ws://localhost:9100")
   await agent.initialize()
   response = await agent.send_message("Walk forward slowly")
   ```

4. **Start OpenPI policy loop** (optional, for learned policy control)
   - The policy loop is started via `policy.start` command through the bridge
   - OpenPI inference ticks are sent as `policy.tick` commands

## Safety Limits

### Motion Bounds (per policy tick)

| Parameter | Min | Max | Unit |
|-----------|-----|-----|------|
| walk_x | -0.05 | 0.05 | meters/step |
| walk_y | -0.05 | 0.05 | meters/step |
| walk_yaw | -10.0 | 10.0 | degrees/step |
| walk_height | 0.015 | 0.06 | meters |
| walk_speed | 1 | 4 | integer |
| head_pan | -1.5 | 1.5 | radians |
| head_tilt | -1.0 | 1.0 | radians |

All policy actions are clamped to these bounds before being sent to the robot.

### Safety Mechanisms

1. **Command rate limiting**: 30 commands/sec per session (configurable)
2. **Deadman heartbeat**: auto-stop if no movement command in 1-2 seconds
3. **Policy heartbeat**: auto-stop policy if no tick in 2 seconds
4. **Motion bound clamping**: all policy actions clamped to safe ranges
5. **Manual preemption**: any manual walk/head/action command auto-stops policy mode
6. **Max step limit**: policy auto-stops after configurable max steps (default 10000)
7. **Disconnect safety**: policy stopped and robot halted on websocket disconnect

## Failure Handling

| Failure | Behavior |
|---------|----------|
| Bridge disconnect | Policy stopped, walk stopped |
| OpenPI inference timeout | Policy tick heartbeat expires, policy stopped |
| Out-of-bounds action | Action clamped, `safety.policy_guard` event with reason |
| Battery low | Telemetry reports low battery; agent should check and stop |
| IMU instability | Visible in telemetry; agent should issue ROBOT_IDLE |
| Policy max steps reached | Policy auto-stopped, `policy.status` event |

## Bridge Protocol Commands

### Existing
- `walk.set` — set walk parameters (x, y, yaw, speed, height)
- `walk.command` — start/stop/enable/disable walking
- `action.play` — play named action sequence
- `head.set` — set head pan/tilt

### Policy Lifecycle
- `policy.start` — start policy loop (requires `task`, optional `hz`, `max_steps`)
- `policy.stop` — stop policy loop (optional `reason`)
- `policy.tick` — apply one policy action (requires `action` dict)
- `policy.status` — query current policy state

### Events
- `telemetry.basic` — robot telemetry (battery, walking, IMU, walk params, head)
- `telemetry.policy` — policy tick telemetry (step, clamped values, guard reason)
- `telemetry.perception` — perception state updates
- `safety.deadman_triggered` — deadman heartbeat timeout fired
- `safety.policy_guard` — policy safety guard triggered (with reason)
- `policy.status` — policy state change (running/idle with reason)

## Eliza Plugin Actions

| Action | Description | Parameters |
|--------|-------------|------------|
| `WALK_COMMAND` | Start/stop walking | `action`: start/stop/enable/disable |
| `WALK_SET` | Set walk direction | `x`, `y`, `yaw`, `speed`, `height` |
| `HEAD_SET` | Move robot head | `pan`, `tilt`, `duration` |
| `ACTION_PLAY` | Play action sequence | `name` |
| `POLICY_START` | Start OpenPI policy | `task`, `hz`, `max_steps` |
| `POLICY_STOP` | Stop policy | `reason` |
| `ROBOT_IDLE` | Emergency stop | (none) |

## Eliza Plugin Providers

| Provider | Description |
|----------|-------------|
| `ROBOT_STATE` | Battery, walking state, IMU, walk params, head position |
| `PERCEPTION_STATE` | Tracked entities, confidence, positions |
| `POLICY_STATE` | Policy active/idle, task, step count, safety events |

## Running the Policy Loop

### Baseline policy (no OpenPI server needed)

```bash
# Start bridge with mock backend
python3 -m bridge.server --backend mock --port 9100 &

# Run baseline forward-walk policy using lifecycle protocol
python3 -m training.runtime.policy_bridge_loop \
  --uri ws://127.0.0.1:9100 --task baseline_forward --hz 5 --steps 50
```

### OpenPI inference loop

```bash
# Start bridge
python3 -m bridge.server --backend mock --port 9100 &

# Run with OpenPI server (passthrough if --openpi-url is empty)
python3 -m training.runtime.openpi_loop \
  --bridge-uri ws://127.0.0.1:9100 \
  --openpi-url http://localhost:8000 \
  --task "walk to the red cup" \
  --hz 5 --max-steps 500 \
  --trace-path traces/openpi_session.jsonl
```

### Passthrough mode (no OpenPI server)

When `--openpi-url` is empty, the loop runs in passthrough mode sending
zero-action ticks. This is useful for testing the full lifecycle without
an inference server.

## Running Tests

```bash
cd /home/shaw/Documents/ainex-robot-code

# All bridge tests (130 tests)
python3 -m pytest bridge/tests/ -v

# Policy-specific tests only
python3 -m pytest bridge/tests/test_policy_protocol.py \
  bridge/tests/test_policy_safety.py \
  bridge/tests/test_openpi_adapter.py \
  bridge/tests/test_perception.py \
  bridge/tests/test_policy_lifecycle.py -v
```
