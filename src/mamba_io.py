"""Save/load helpers for mamba-ssm MambaLMHeadModel checkpoints.

mamba-ssm doesn't ship a save_pretrained equivalent, so we write the
state dict and config to disk in a layout that mirrors a HF repo
(pytorch_model.bin + config.json) and reload them by reconstructing
the model from MambaConfig.
"""

import json
from dataclasses import asdict
from pathlib import Path

import torch
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel


def save_mamba(model: MambaLMHeadModel, save_dir) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_dir / "pytorch_model.bin")
    with open(save_dir / "config.json", "w") as f:
        json.dump(asdict(model.config), f, indent=2)


def load_mamba(checkpoint_dir, device, dtype) -> MambaLMHeadModel:
    checkpoint_dir = Path(checkpoint_dir)
    with open(checkpoint_dir / "config.json") as f:
        config = MambaConfig(**json.load(f))
    model = MambaLMHeadModel(config, device=device, dtype=dtype)
    state = torch.load(checkpoint_dir / "pytorch_model.bin", map_location=device)
    model.load_state_dict(state)
    return model