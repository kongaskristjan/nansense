"""Suite-wide test configuration.

The suite runs under `pytest-xdist` by default (see `addopts` in
pyproject.toml), which changes what "fast" means: the work is spread over
worker processes, so the thing to avoid is each worker fighting the others
for cores.
"""

from __future__ import annotations

import os

import pytest
import torch

# Threads each xdist worker may use. Torch otherwise sizes its intra-op pool
# from the machine's core count, which is right for one process and badly
# wrong for ten: the tensors here are small enough that the pool wins little,
# while every worker oversubscribing the box costs roughly five times the CPU
# for the same wall clock. Two measured fastest — enough to keep the few
# convolution-heavy tests off a single core, low enough that the workers
# together still fit the machine.
_WORKER_THREADS = 2


def pytest_configure(config: pytest.Config) -> None:
    """Cap torch's thread pool inside xdist workers. A serial run keeps the
    default, where having the whole machine to itself is the right call."""
    if os.environ.get("PYTEST_XDIST_WORKER"):
        torch.set_num_threads(_WORKER_THREADS)
