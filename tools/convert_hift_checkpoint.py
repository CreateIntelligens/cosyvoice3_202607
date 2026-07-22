#!/usr/bin/env python3
"""Convert between inference HiFT and training HiFi-GAN checkpoints."""

import argparse
from pathlib import Path

import torch


METADATA_KEYS = {"epoch", "step"}


def tensor_state(path: Path) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError(f"{path} does not contain a state-dict mapping")
    return {
        key: value
        for key, value in state.items()
        if key not in METADATA_KEYS and isinstance(key, str) and torch.is_tensor(value)
    }


def wrap(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if any(key.startswith("discriminator.") for key in state):
        raise ValueError("Input contains discriminator weights; use a bare inference hift.pt")
    if any(key.startswith("generator.") for key in state):
        result = {key: value for key, value in state.items() if key.startswith("generator.")}
    else:
        result = {f"generator.{key}": value for key, value in state.items()}
    if not result:
        raise ValueError("No generator tensors found")
    return result


def extract(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    result = {
        key.removeprefix("generator."): value
        for key, value in state.items()
        if key.startswith("generator.")
    }
    if not result:
        raise ValueError("No generator.* tensors found in the training checkpoint")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("wrap", "extract"))
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()

    if args.source.resolve() == args.destination.resolve():
        raise ValueError("Source and destination must be different; never overwrite official hift.pt")

    source_state = tensor_state(args.source)
    output_state = wrap(source_state) if args.mode == "wrap" else extract(source_state)
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_state, args.destination)

    tensor_count = len(output_state)
    parameter_count = sum(value.numel() for value in output_state.values())
    print(f"mode={args.mode} tensors={tensor_count} parameters={parameter_count} output={args.destination}")


if __name__ == "__main__":
    main()
