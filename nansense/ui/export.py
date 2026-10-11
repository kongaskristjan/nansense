"""The Download buttons: one layer's snapshot tensors as a `.pt` file.

The UI side of `nansense.export` (the agent side is the `export_tensors` MCP
tool). A layer card downloads the layer's activation, gradient and custom
tensors; the weights page downloads the layer's parameters with their
gradients and optimizer state. Both fetch the same route, which builds the file
from whatever snapshot is published when the request arrives.

The click handler builds the selection first (cheap: references into the
snapshot, no copy) so a refusal — nothing captured yet, over the size cap —
shows as a notification instead of a failed browser download.
"""

from __future__ import annotations

from urllib.parse import urlencode

from fastapi import FastAPI, HTTPException, Response
from nicegui import ui

from nansense.export import ExportError, build_export
from nansense.session import Session

EXPORT_PATH: str = "/nansense/export.pt"


def export_href(layer: str, *, activations: bool, weights: bool) -> str:
    query = urlencode(
        {"layer": layer, "activations": int(activations), "weights": int(weights)}
    )
    return f"{EXPORT_PATH}?{query}"


def add_export_route(app: FastAPI, session: Session) -> None:
    """Serve `export_href` downloads; registered by `ui.app.serve` ahead of
    NiceGUI's catch-all mount. A sync handler, so Starlette runs the
    serialization in its threadpool rather than on the event loop."""

    @app.get(EXPORT_PATH, include_in_schema=False)
    def download_export(layer: str, activations: int = 1, weights: int = 1) -> Response:
        try:
            built = build_export(
                session,
                layers=[layer],
                activations=bool(activations),
                weights=bool(weights),
            )
        except ExportError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return Response(
            built.to_bytes(),
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f'attachment; filename="{built.filename}"'
            },
        )


def start_download(
    session: Session, layer: str, *, activations: bool, weights: bool
) -> None:
    """Click handler: notify why not, or hand the browser the download URL."""
    try:
        build_export(
            session, layers=[layer], activations=activations, weights=weights
        )
    except ExportError as exc:
        ui.notify(str(exc), type="warning")
        return
    ui.download.from_url(
        export_href(layer, activations=activations, weights=weights)
    )
