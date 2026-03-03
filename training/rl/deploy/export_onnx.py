"""Export trained checkpoints to ONNX for deployment on Raspberry Pi.

Usage:
    python -m training.rl.deploy.export_onnx \
        --checkpoint checkpoints/phase1_locomotion/model_01500.pt \
        --output models/ainex_locomotion.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import numpy as np

OBS_DIM = 48


class ActorOnly(torch.nn.Module):
    """Extract just the actor network for deployment (no critic needed)."""

    def __init__(self, actor: torch.nn.Module):
        super().__init__()
        self.actor = actor

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(obs)


def export_locomotion_onnx(
    checkpoint_path: str,
    output_path: str,
    opset_version: int = 11,
) -> None:
    """Export locomotion policy to ONNX.

    Args:
        checkpoint_path: Path to .pt checkpoint.
        output_path: Output .onnx file path.
        opset_version: ONNX opset version.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})

    # Expect the checkpoint to contain a full model object or state_dict.
    if "model" not in ckpt:
        raise ValueError("Checkpoint does not contain 'model' key")

    model = ckpt["model"]
    if isinstance(model, dict):
        raise ValueError(
            "Checkpoint contains a state_dict but no model class. "
            "Cannot reconstruct network architecture for ONNX export."
        )

    model.eval()

    # Extract actor only.
    if hasattr(model, "actor"):
        actor = ActorOnly(model.actor)
    else:
        actor = model
    actor.eval()

    obs_dim = config.get("obs_dim", OBS_DIM)
    action_dim = config.get("action_dim", 12)

    # Dummy input.
    dummy = torch.randn(1, obs_dim)

    # Export.
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        actor,
        dummy,
        str(output),
        opset_version=opset_version,
        input_names=["observation"],
        output_names=["action"],
        dynamic_axes={
            "observation": {0: "batch"},
            "action": {0: "batch"},
        },
    )

    print(f"Exported ONNX model to {output}")
    print(f"  Input: observation [{obs_dim}]")
    print(f"  Output: action [{action_dim}]")

    # Validate.
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(output))
        result = sess.run(None, {"observation": dummy.numpy()})
        print(f"  ONNX validation: OK (output shape {result[0].shape})")
    except ImportError:
        print("  (onnxruntime not installed, skipping validation)")


def export_meta_onnx(
    checkpoint_path: str,
    output_path: str,
    opset_version: int = 11,
) -> None:
    """Export meta-policy to ONNX."""
    from training.rl.meta.meta_policy import MetaPolicyNetwork, EMBEDDING_DIM, ROBOT_STATE_DIM

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    skill_names = ckpt.get("skill_names", ["stand", "walk", "turn", "wave", "bow"])
    model = MetaPolicyNetwork(num_skills=len(skill_names))
    model.load_state_dict(ckpt["model"])
    model.eval()

    # For ONNX, combine text_emb and state into a single input.
    class MetaONNX(torch.nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, x):
            text_emb = x[:, :EMBEDDING_DIM]
            state = x[:, EMBEDDING_DIM:]
            logits, params = self.net(text_emb, state)
            return logits, params

    wrapper = MetaONNX(model)
    dummy = torch.randn(1, EMBEDDING_DIM + ROBOT_STATE_DIM)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapper,
        dummy,
        str(output),
        opset_version=opset_version,
        input_names=["input"],
        output_names=["skill_logits", "params"],
        dynamic_axes={"input": {0: "batch"}, "skill_logits": {0: "batch"}, "params": {0: "batch"}},
    )

    print(f"Exported meta-policy ONNX to {output}")


def main():
    parser = argparse.ArgumentParser(description="Export RL checkpoint to ONNX")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--output", required=True, help="Output .onnx path")
    parser.add_argument("--type", choices=["locomotion", "meta"], default="locomotion")
    parser.add_argument("--opset", type=int, default=11)
    args = parser.parse_args()

    if args.type == "locomotion":
        export_locomotion_onnx(args.checkpoint, args.output, args.opset)
    else:
        export_meta_onnx(args.checkpoint, args.output, args.opset)


if __name__ == "__main__":
    main()
