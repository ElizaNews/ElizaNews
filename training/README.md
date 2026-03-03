# Training Workspace

This workspace holds artifacts for policy-vector and locomotion training.

## Scope

- behavior cloning from scripted/action-group trajectories
- reinforcement learning fine-tuning in simulation
- embedding-to-policy-vector adapters for agent intent conditioning
- model export for runtime bridge execution

## Planned Layout

- `training/interfaces.py`: typed runtime interfaces for policy inference
- `training/configs/policy_vector_baseline.json`: starter hyperparameter config
- `training/datasets/`: trajectory exports and metadata
- `training/configs/`: experiment configs
- `training/models/`: checkpoints
- `training/eval/`: parity and safety evaluation scripts
- `training/runtime/policy_bridge_loop.py`: baseline policy loop that emits bridge commands
- `training/datasets/build_from_trace.py`: converter from bridge trace logs to supervised dataset JSONL

## Initial Principle

The runtime bridge API is the contract boundary. Training code can change internals freely, but exported policy runtime must map onto bridge command primitives.

