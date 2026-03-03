# AiNex Bridge Runbook

This runbook provides concrete command sequences for real robot mode, sim mode, and Isaac preparation.

## 1) Bridge Environment

```bash
cd /home/shaw/Documents/ainex-robot-code
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r bridge/requirements.txt
```

## 2) Real Robot Bringup + Bridge

In your ROS environment (where AiNex packages resolve):

```bash
roslaunch ainex_bringup bringup.launch
```

Then start bridge:

```bash
cd /home/shaw/Documents/ainex-robot-code
source .venv/bin/activate
PYTHONPATH=. ./bridge/scripts/start_bridge_real.sh
```

Optional trace logging:

```bash
TRACE_LOG_PATH=/tmp/ainex_bridge_trace.jsonl PYTHONPATH=. ./bridge/scripts/start_bridge_real.sh
```

## 3) Gazebo Sim Bringup + Bridge

Terminal A:

```bash
./bridge/scripts/start_ros_sim_stack.sh
```

Terminal B:

```bash
./bridge/scripts/start_ros_sim_controller.sh
```

Terminal C:

```bash
cd /home/shaw/Documents/ainex-robot-code
source .venv/bin/activate
PYTHONPATH=. ./bridge/scripts/start_bridge_sim.sh
```

## 4) Replay Test

```bash
cd /home/shaw/Documents/ainex-robot-code
source .venv/bin/activate
PYTHONPATH=. python -m bridge.tools.replay_trace \
  --uri ws://127.0.0.1:9100 \
  --trace bridge/examples/basic_walk_trace.jsonl
```

## 4b) ROSBridge-Compatible Mode

If your client framework expects ROSBridge op messages, run:

```bash
cd /home/shaw/Documents/ainex-robot-code
source .venv/bin/activate
PYTHONPATH=. python -m bridge.rosbridge_server --backend mock --port 9091
```

Then use `op=subscribe|publish|call_service` style payloads.

## 5) Real-vs-Sim Parity Check

If real bridge is on `9101` and sim bridge is on `9102`:

```bash
cd /home/shaw/Documents/ainex-robot-code
source .venv/bin/activate
PYTHONPATH=. python -m bridge.tools.parity_check \
  --left-uri ws://127.0.0.1:9101 \
  --right-uri ws://127.0.0.1:9102 \
  --trace bridge/examples/basic_walk_trace.jsonl
```

## 6) Isaac Integration Checklist

The `isaac` backend is scaffolded but not yet connected to Isaac runtime. Complete these steps:

1. Install NVIDIA driver and CUDA supported by your Isaac stack.
2. Install Isaac runtime and Python API in a dedicated env.
3. Import/export AiNex articulation (URDF/xacro -> Isaac articulation).
4. Implement `bridge/backends/isaac_backend.py` to map:
   - `walk.set`
   - `walk.command`
   - `action.play`
   - `head.set`
   - telemetry events (`imu`, `is_walking`, camera metadata).
5. Add domain randomization config and evaluation scenes.
6. Add parity trace tests against `ros_sim`.

## 7) Training Checklist

1. Build trajectory export job from bridge command+telemetry logs.
2. Implement behavior cloning baseline.
3. Add policy-vector conditioned runtime over `training/interfaces.py`.
4. Add RL fine-tuning tasks in Isaac scenes.
5. Add safety gating for deployment.

## 8) Baseline Policy Runtime Loop

Run a simple policy loop against the bridge:

```bash
cd /home/shaw/Documents/ainex-robot-code
source .venv/bin/activate
PYTHONPATH=. python -m training.runtime.policy_bridge_loop \
  --uri ws://127.0.0.1:9100 \
  --hz 5 \
  --steps 50
```

## 9) Build Dataset From Trace Log

If bridge was started with `--trace-log-path`:

```bash
cd /home/shaw/Documents/ainex-robot-code
source .venv/bin/activate
python training/datasets/build_from_trace.py \
  --input-trace /tmp/ainex_bridge_trace.jsonl \
  --output-jsonl /tmp/ainex_dataset.jsonl
```

