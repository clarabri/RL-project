"""Tests for scripts/update_algo_results.py (no W&B network calls)."""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts.update_algo_results import (
    ExperimentSpec,
    ResultRow,
    build_table,
    format_project_not_found_error,
    infer_experiment_config,
    load_experiment_registry,
    replace_results_table,
    resolve_entity,
    row_to_markdown,
)


@pytest.fixture
def registry() -> list[ExperimentSpec]:
    return load_experiment_registry()


def test_load_experiment_registry_includes_known_experiments(registry):
    paths = {spec.path for spec in registry}
    assert "dqn/cartpole" in paths
    assert "dqn/pong" in paths
    assert "ddpg/halfcheetah" in paths
    assert "a2c/halfcheetah" in paths


def test_infer_experiment_config_cartpole(registry):
    config = {
        "algorithm": {"_target_": "src.algorithms.dqn.DQNAlgorithm"},
        "environment": {"name": "CartPole-v1"},
        "trainer": {"seed": 42, "total_frames": 500_100},
    }
    assert infer_experiment_config(config, registry) == "experiment=dqn/cartpole"


def test_infer_experiment_config_pong(registry):
    config = {
        "algorithm": {
            "_target_": "src.algorithms.dqn.DQNAlgorithm",
            "obs_key": "pixels",
        },
        "environment": {"name": "ALE/Pong-v5"},
        "trainer": {"seed": 42, "total_frames": 40_000_100},
    }
    assert infer_experiment_config(config, registry) == "experiment=dqn/pong"


def test_infer_experiment_config_halfcheetah_ddpg(registry):
    config = {
        "algorithm": {"_target_": "src.algorithms.ddpg.DDPGAlgorithm"},
        "environment": {"name": "HalfCheetah-v4"},
        "trainer": {"seed": 42, "total_frames": 1_000_000},
    }
    assert infer_experiment_config(config, registry) == "experiment=ddpg/halfcheetah"


def test_build_table_empty():
    table = build_table([])
    assert "No finished runs tagged ``template`` yet" in table
    assert "| Run | Environment |" in table


def test_row_to_markdown():
    row = ResultRow(
        run_name="dqn_cartpole_2025-01-01",
        run_url="https://wandb.ai/LatentLab/torchrl-hydra-template/runs/abc123",
        environment="CartPole-v1",
        config="experiment=dqn/cartpole",
        seed=42,
        frames=500_100,
        eval_return="500.0",
        notes="—",
    )
    md = row_to_markdown(row)
    assert "[dqn_cartpole_2025-01-01](" in md
    assert "`experiment=dqn/cartpole`" in md
    assert "500,100" in md


def test_replace_results_table():
    readme = Path("src/algorithms/dqn/README.md").read_text(encoding="utf-8")
    new_table = build_table([])
    updated = replace_results_table(readme, new_table)
    assert updated != readme
    assert "No finished runs tagged ``template`` yet" in updated
    assert "## Experimental results" in updated


def test_resolve_entity_prefers_explicit():
    class FakeApi:
        default_entity = "login-default"

    assert resolve_entity(FakeApi(), "explicit") == "explicit"


def test_resolve_entity_uses_env(monkeypatch):
    class FakeApi:
        default_entity = "login-default"

    monkeypatch.setenv("WANDB_ENTITY", "from-env")
    assert resolve_entity(FakeApi(), None) == "from-env"


def test_format_project_not_found_error_lists_projects():
    msg = format_project_not_found_error(
        entity="rschwinger",
        project="torchrl-hydra-template",
        available_projects=["introdrl", "PretrainWM"],
    )
    assert "rschwinger/torchrl-hydra-template" in msg
    assert "introdrl" in msg
    assert "LatentLab" in msg
