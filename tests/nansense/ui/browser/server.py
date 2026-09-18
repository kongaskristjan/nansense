"""Tiny deterministic training loop for the optional browser smoke suite."""

from __future__ import annotations

import argparse

import torch
from torch import nn

import nansense
from examples.custom_metrics.main import BlobNet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(7)
    model = BlobNet(channels=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    session = nansense.start(model, epochs=1, phases={"train": 20}, optimizer=optimizer)
    session.set_experiment_defaults(steps=2, channels=2, jitter=0)
    session.watch("conv1")
    session.watch("conv2")
    nansense.serve(session, port=args.port, open_browser=False)
    inputs = torch.rand(4, 1, 8, 8)
    labels = torch.tensor([0, 1, 2, 3])
    try:
        for _ in range(20):
            with session.batch(phase="train", epoch=0):
                optimizer.zero_grad(set_to_none=True)
                nn.functional.cross_entropy(model(inputs), labels).backward()
                optimizer.step()
    finally:
        session.close()


if __name__ == "__main__":
    main()
