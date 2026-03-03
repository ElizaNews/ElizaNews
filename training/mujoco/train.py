"""Train AiNex locomotion policy using MuJoCo Playground + Brax PPO.

GPU-accelerated training using MJX for physics and JAX for RL.
Following the Playground training pattern for zero-shot sim-to-real.

Usage:
    python3 -m training.mujoco.train
    python3 -m training.mujoco.train --target
    python3 -m training.mujoco.train --num-timesteps 50000000
    python3 -m training.mujoco.train --no-domain-rand
"""

import argparse
import functools
import json
import time
from datetime import datetime
from pathlib import Path

import jax

from training.mujoco.joystick import Joystick
from training.mujoco.joystick import default_config as joystick_default_config
from training.mujoco.target import TargetReaching
from training.mujoco.target import default_config as target_default_config
from training.mujoco.randomize import domain_randomize


def make_ppo_config(num_timesteps: int = 100_000_000) -> dict:
    """PPO hyperparameters following Playground locomotion defaults."""
    return {
        "num_timesteps": num_timesteps,
        "num_evals": 10,
        "reward_scaling": 1.0,
        "normalize_observations": True,
        "action_repeat": 1,
        "unroll_length": 20,
        "num_minibatches": 32,
        "num_updates_per_batch": 4,
        "discounting": 0.97,
        "learning_rate": 3e-4,
        "entropy_cost": 1e-2,
        "num_envs": 4096,
        "batch_size": 256,
        "max_grad_norm": 1.0,
        "policy_hidden_layer_sizes": (128, 128, 128, 128),
        "value_hidden_layer_sizes": (256, 256, 256, 256, 256),
    }


def train(
    num_timesteps: int = 100_000_000,
    num_envs: int = 4096,
    checkpoint_dir: str = "checkpoints/mujoco_locomotion",
    seed: int = 0,
    target: bool = False,
    domain_rand: bool = True,
    enable_entity_slots: bool = False,
):
    """Run GPU-accelerated PPO training for AiNex locomotion."""
    from brax.training.agents.ppo.train import train as ppo_train
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.io import model as brax_model
    from mujoco_playground._src.wrapper import wrap_for_brax_training

    print("=" * 60)
    print("AiNex MuJoCo Playground Training")
    print("=" * 60)
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
    print()

    # Create environment
    if target:
        env_config = target_default_config()
        env_name = "AiNexTargetReaching"
    else:
        env_config = joystick_default_config()
        env_name = "AiNexJoystick"
    env_config.enable_entity_slots = enable_entity_slots
    if target:
        env = TargetReaching(config=env_config)
    else:
        env = Joystick(config=env_config)
    print(f"Environment: {env_name}")
    print(f"  Action size: {env.action_size}")
    print(f"  MuJoCo model: nq={env.mj_model.nq}, nv={env.mj_model.nv}, nu={env.mj_model.nu}")
    print(f"  Sim dt: {env_config.sim_dt}, Ctrl dt: {env_config.ctrl_dt}")
    print(f"  Episode length: {env_config.episode_length}")
    if enable_entity_slots:
        print(f"  Entity slots: ENABLED (152 extra obs dims, {len(env._entity_body_ids)} entities)")
    print()

    # PPO config
    ppo_cfg = make_ppo_config(num_timesteps)
    ppo_cfg["num_envs"] = num_envs
    print(f"PPO Config:")
    for k, v in ppo_cfg.items():
        print(f"  {k}: {v}")
    print()

    # Domain randomization
    rand_fn = None
    if domain_rand:
        print("Domain randomization: ENABLED")
        rand_fn = domain_randomize
        print(f"  Randomizing: friction, mass, armature, damping, gains, qpos0")
    else:
        print("Domain randomization: DISABLED")
    print()

    # Checkpoint directory
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Compute and store obs_size for checkpoint compatibility
    obs_size = env._config.obs_history_size * env._single_obs_size
    if enable_entity_slots:
        from perception.entity_slots.slot_config import TOTAL_ENTITY_DIMS
        obs_size += TOTAL_ENTITY_DIMS

    # Save config
    with open(ckpt_dir / "config.json", "w") as f:
        json.dump({
            "env": env_name,
            "ppo": ppo_cfg,
            "env_config": {k: str(v) for k, v in dict(env_config).items()},
            "domain_rand": domain_rand,
            "enable_entity_slots": enable_entity_slots,
            "obs_size": obs_size,
            "action_size": env.action_size,
            "seed": seed,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2, default=str)

    # Training metrics tracking
    train_metrics = []
    best_reward = float("-inf")
    start_time = time.time()

    def progress_callback(num_steps, metrics):
        nonlocal best_reward
        elapsed = time.time() - start_time
        reward = float(metrics.get("eval/episode_reward", 0))
        train_metrics.append({
            "steps": int(num_steps),
            "reward": reward,
            "elapsed": elapsed,
        })

        print(f"Step {num_steps:>10,} | "
              f"Reward: {reward:>8.2f} | "
              f"Time: {elapsed:>6.1f}s | "
              f"FPS: {num_steps/max(elapsed,1):.0f}",
              flush=True)

        if reward > best_reward:
            best_reward = reward
            print(f"  New best reward: {reward:.2f}", flush=True)

    # Run PPO training
    print("Starting training...")
    print()

    # Let Brax PPO handle wrapping — it creates separate train/eval envs
    # with correct batch sizes for domain randomization
    make_inference_fn, params, _ = ppo_train(
        environment=env,
        num_timesteps=ppo_cfg["num_timesteps"],
        episode_length=env_config.episode_length,
        num_evals=ppo_cfg["num_evals"],
        reward_scaling=ppo_cfg["reward_scaling"],
        normalize_observations=ppo_cfg["normalize_observations"],
        action_repeat=ppo_cfg["action_repeat"],
        unroll_length=ppo_cfg["unroll_length"],
        num_minibatches=ppo_cfg["num_minibatches"],
        num_updates_per_batch=ppo_cfg["num_updates_per_batch"],
        discounting=ppo_cfg["discounting"],
        learning_rate=ppo_cfg["learning_rate"],
        entropy_cost=ppo_cfg["entropy_cost"],
        num_envs=ppo_cfg["num_envs"],
        batch_size=ppo_cfg["batch_size"],
        max_grad_norm=ppo_cfg["max_grad_norm"],
        wrap_env=True,
        wrap_env_fn=functools.partial(
            wrap_for_brax_training,
        ),
        randomization_fn=rand_fn,
        network_factory=lambda obs_size, action_size, preprocess_observations_fn: ppo_networks.make_ppo_networks(
            obs_size,
            action_size,
            preprocess_observations_fn=preprocess_observations_fn,
            policy_hidden_layer_sizes=ppo_cfg["policy_hidden_layer_sizes"],
            value_hidden_layer_sizes=ppo_cfg["value_hidden_layer_sizes"],
        ),
        seed=seed,
        progress_fn=progress_callback,
    )

    elapsed = time.time() - start_time
    print()
    print(f"Training complete in {elapsed:.1f}s")
    print(f"Best reward: {best_reward:.2f}")

    # Save final checkpoint
    final_path = ckpt_dir / "final_params"
    brax_model.save_params(final_path, params)
    print(f"Saved params: {final_path}")

    # Save training metrics
    with open(ckpt_dir / "metrics.json", "w") as f:
        json.dump(train_metrics, f, indent=2)

    # Save inference function for deployment
    inference_fn = make_inference_fn(params, deterministic=True)
    print(f"\nCheckpoints saved to {ckpt_dir}/")

    return inference_fn, params


def main():
    parser = argparse.ArgumentParser(description="Train AiNex locomotion with MuJoCo Playground")
    parser.add_argument("--num-timesteps", type=int, default=100_000_000,
                        help="Total training timesteps")
    parser.add_argument("--num-envs", type=int, default=4096,
                        help="Number of parallel environments")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/mujoco_locomotion",
                        help="Checkpoint output directory")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target", action="store_true",
                        help="Use target-reaching task instead of velocity tracking")
    parser.add_argument("--no-domain-rand", action="store_true",
                        help="Disable domain randomization")
    parser.add_argument("--enable-entity-slots", action="store_true",
                        help="Include entity perception slots in observations (152 extra dims)")
    args = parser.parse_args()

    # Default checkpoint dir changes for target task
    ckpt_dir = args.checkpoint_dir
    if args.target and ckpt_dir == "checkpoints/mujoco_locomotion":
        ckpt_dir = "checkpoints/mujoco_target"

    train(
        num_timesteps=args.num_timesteps,
        num_envs=args.num_envs,
        checkpoint_dir=ckpt_dir,
        seed=args.seed,
        target=args.target,
        domain_rand=not args.no_domain_rand,
        enable_entity_slots=args.enable_entity_slots,
    )


if __name__ == "__main__":
    main()
