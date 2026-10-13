"""Judge tooling — schema shape and median/representative math (no network)."""

from __future__ import annotations

from typing import Any

from scripts.judge_run import (
    _AXES,
    JUDGE_SCHEMA,
    median_axes,
    render_scorecard,
    representative_pass,
)


def _pass(scores: dict[str, int], **extra: Any) -> dict[str, Any]:
    full = {name: scores.get(name, 0) for name, _ in _AXES}
    return {
        "scores": full,
        "violations": [],
        "claims_ledger": [],
        "summary": "fine",
        **extra,
    }


def test_schema_is_flat_and_requires_every_axis() -> None:
    # Flat by necessity: the nested per-axis version exceeded the
    # structured-output grammar-size limit (verified live 2026-07-11).
    scores = JUDGE_SCHEMA["properties"]["scores"]
    assert set(scores["required"]) == {name for name, _ in _AXES}
    assert scores["additionalProperties"] is False
    violation_props = JUDGE_SCHEMA["properties"]["violations"]["items"]["properties"]
    assert set(violation_props["attribution"]["enum"]) == {"model", "carry", "event_engine"}
    assert set(violation_props["axis"]["enum"]) == {name for name, _ in _AXES}

    # No numerical constraints anywhere (unsupported by the API subset).
    def walk(node: Any) -> None:
        if isinstance(node, dict):
            assert "minimum" not in node and "maximum" not in node
            for value in node.values():
                walk(value)

    walk(JUDGE_SCHEMA)


def test_median_axes_takes_the_middle_pass_and_clamps() -> None:
    passes = [
        _pass({"fact_continuity": 10, "plot_progression": 5}),
        _pass({"fact_continuity": 20, "plot_progression": 8}),
        _pass({"fact_continuity": 99, "plot_progression": 11}),  # 99 clamps to 25
    ]
    medians = median_axes(passes)
    assert medians["fact_continuity"] == 20
    assert medians["plot_progression"] == 8
    assert medians["prose_variety"] == 0


def test_representative_pass_is_closest_to_medians() -> None:
    near = _pass({"fact_continuity": 20})
    far = _pass({"fact_continuity": 2})
    medians = median_axes([near, far, _pass({"fact_continuity": 19})])
    rep = representative_pass([near, far], medians)
    assert rep is near


def test_render_scorecard_carries_headline_and_violations(tmp_path: Any) -> None:
    rep = _pass(
        {"fact_continuity": 20},
        violations=[
            {
                "axis": "fact_continuity",
                "turn": 14,
                "claim": "Marta is in two places",
                "evidence_quote": "she waves from the gate",
                "attribution": "carry",
            }
        ],
    )
    medians = {name: mx for name, mx in _AXES}
    card = render_scorecard(
        replay_dir=tmp_path,
        det_total=88,
        det_max=100,
        medians=medians,
        rep=rep,
        n_passes=3,
    )
    assert "D = 88/100" in card
    assert "J = 100/100" in card
    assert "combined 188/200" in card
    assert "[fact_continuity/carry] Marta is in two places" in card
