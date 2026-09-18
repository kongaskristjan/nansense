"""Validate external parameters before they become queued, immutable requests."""

from dataclasses import fields

import pytest

from nansense.contracts.experiments import DreamParams, parse_params, resolve_params
from nansense.experiments import EXPERIMENT_PARAMS
from tests.nansense.helpers import make_session


@pytest.mark.parametrize("kind", list(EXPERIMENT_PARAMS))
def test_form_schema_matches_the_typed_request(kind: str) -> None:
    params = parse_params(kind, {})
    assert params.kind == kind
    assert {f.name for f in fields(params)} - {"mean", "std"} == {
        spec.key for spec in EXPERIMENT_PARAMS[kind]
    }
    for spec in EXPERIMENT_PARAMS[kind]:
        assert getattr(params, spec.key) == spec.default


@pytest.mark.parametrize(
    "bad",
    [
        {"stepps": 5},
        {"steps": "five"},
        {"steps": float("nan")},
        {"lr": float("inf")},
        {"clamp": "false"},
        {"start": "random"},
        {"mean": [float("nan")]},
        {"std": [0.0]},
    ],
)
def test_invalid_replacement_leaves_the_existing_request_queued(
    bad: dict[str, object],
) -> None:
    session, _ = make_session()
    first = session.register_auto_experiment(
        "tab",
        kind="deep_dream",
        layer="fc1",
        params={"steps": 1},
    )
    with pytest.raises(ValueError):
        session.register_auto_experiment(
            "tab", kind="deep_dream", layer="fc1", params=bad
        )
    assert session.experiment_queue_state(first).stage == "queued"
    assert (
        session.request_experiment(kind="deep_dream", layer="fc1", params={})
        == first + 1
    )


def test_parameter_resolution_and_freezing_preserve_the_boundary() -> None:
    mean = [0.2, 0.3, 0.4]
    external, ignored = resolve_params(
        "deep_dream",
        {"steps": 1000, "mean": mean, "foreign": True},
        {"steps": 2, "channels": 4},
    )
    parsed = parse_params("deep_dream", external, locked=True)
    assert isinstance(parsed, DreamParams)
    assert parsed.steps == 300 and parsed.channels == 4
    assert ignored == ["foreign"]
    mean[0] = 10.0
    external["steps"] = 1
    assert parsed.mean == (0.2, 0.3, 0.4) and parsed.steps == 300
