"""Import boundaries must hold in a fresh process, before test collection loads UI."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path


def test_headless_use_and_rendering_do_not_load_the_web_server(tmp_path: Path) -> None:
    script = """
        import importlib.abc
        import logging
        import sys
        import warnings
        from types import SimpleNamespace

        class BlockWebImports(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname: str, path: object = None, target: object = None) -> None:
                if fullname.split(".")[0] in {"nicegui", "uvicorn", "fastapi", "mcp"}:
                    raise AssertionError(f"headless usage imported {fullname}")

        sys.meta_path.insert(0, BlockWebImports())
        import torch
        import nansense
        import nansense.ui

        assert "nansense.ui.app" not in sys.modules
        assert "nansense.ui.render" not in sys.modules
        assert "nansense.ui.graph" not in sys.modules
        assert "captum" not in sys.modules
        assert "matplotlib" not in sys.modules

        model = torch.nn.Linear(2, 1)
        session = nansense.start(model, epochs=1, phases={"train": 1}, enabled=False,
                                 port=8080, open_browser=False)
        assert nansense.serve(session) is None
        assert nansense.ui.serve(SimpleNamespace(enabled=True, is_leader=False)) is None
        with session.batch(phase="train", epoch=0):
            model(torch.ones(1, 2)).sum().backward()
        assert model.weight.grad is not None

        session = nansense.start(model, epochs=1, phases={"train": 1})
        session.detach()
        with session.batch(phase="train", epoch=0):
            model(torch.ones(1, 2)).sum().backward()
        session.close()

        from nansense.ui import RenderOptions, build_mermaid, render_image, render_strip
        assert "graph" in build_mermaid(model) or "flowchart" in build_mermaid(model)
        assert render_image(torch.zeros(1, 1, 2, 2), 0)
        assert render_strip(torch.zeros(1, 2), 0, options=RenderOptions()).tiles
        assert set(nansense.ui.__all__) <= set(dir(nansense.ui))
        assert "nansense.ui.app" not in sys.modules
        assert not logging.getLogger("nicegui").filters
        assert not any("reduce_op" in str(item[1]) for item in warnings.filters)
        sys.meta_path.pop(0)
        import nansense.ui.app
        assert not logging.getLogger("nicegui").filters
        assert not any("reduce_op" in str(item[1]) for item in warnings.filters)
    """
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        env={**os.environ, "MPLCONFIGDIR": str(tmp_path)},
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
