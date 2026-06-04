#!/usr/bin/env python3
"""Populate algorithm README experimental-results tables from W&B runs.

Fetches finished runs tagged ``template`` from the torchrl-hydra-template W&B
project and rewrites the markdown table under ``## Experimental results`` in
each algorithm README.

Usage::

    # Requires ``wandb login`` (or WANDB_API_KEY).
    python scripts/update_algo_results.py

    # Preview without writing files.
    python scripts/update_algo_results.py --dry-run

    # Override project / entity / tag.
    python scripts/update_algo_results.py --entity LatentLab --tag template
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
def algo_readme_path(algo: str) -> Path:
    return REPO_ROOT / "src" / "algorithms" / algo / "README.md"
EXPERIMENT_DIR = REPO_ROOT / "configs" / "experiment"
WANDB_TABLE_URL = "https://wandb.ai/LatentLab/torchrl-hydra-template/table"
CANONICAL_ENTITY = "LatentLab"
DEFAULT_PROJECT = "torchrl-hydra-template"

TABLE_HEADER = (
    "| Run | Environment | Config | Seed | Frames | Eval return | Notes |"
)
TABLE_SEPARATOR = (
    "|-----|-------------|--------|------|--------|-------------|-------|"
)

# Maps ``src.algorithms.<pkg>.<Class>`` prefix to README package directory.
ALGO_TARGET_PREFIXES: dict[str, str] = {
    "src.algorithms.dqn.": "dqn",
    "src.algorithms.ddpg.": "ddpg",
    "src.algorithms.a2c.": "a2c",
}


@dataclass(frozen=True)
class ExperimentSpec:
    """One composed experiment under ``configs/experiment/``."""

    path: str  # e.g. ``dqn/cartpole``
    algorithm_choice: str  # e.g. ``dqn``, ``dqn_atari``
    environment_choice: str  # e.g. ``cartpole``, ``pong_train``


@dataclass(frozen=True)
class ResultRow:
    """One row in an algorithm README results table."""

    run_name: str
    run_url: str
    environment: str
    config: str
    seed: int | None
    frames: int | None
    eval_return: str
    notes: str


def load_experiment_registry(root: Path = EXPERIMENT_DIR) -> list[ExperimentSpec]:
    """Parse ``configs/experiment/**/*.yaml`` defaults into experiment specs."""
    specs: list[ExperimentSpec] = []
    for path in sorted(root.rglob("*.yaml")):
        rel = path.relative_to(root)
        exp_path = rel.with_suffix("").as_posix()
        text = path.read_text(encoding="utf-8")
        algo = _parse_override(text, "/algorithm")
        env = _parse_override(text, "/environment")
        if algo is None or env is None:
            continue
        specs.append(
            ExperimentSpec(
                path=exp_path,
                algorithm_choice=algo,
                environment_choice=env,
            )
        )
    return specs


def _parse_override(text: str, group: str) -> str | None:
    match = re.search(rf"override {re.escape(group)}:\s*(\S+)", text)
    return match.group(1) if match else None


def algo_package_from_target(target: str | None) -> str | None:
    if not target:
        return None
    for prefix, package in ALGO_TARGET_PREFIXES.items():
        if target.startswith(prefix):
            return package
    return None


def infer_experiment_config(
    config: dict,
    registry: list[ExperimentSpec],
) -> str:
    """Return ``experiment=<path>`` for a W&B run config."""
    explicit = config.get("experiment")
    if explicit not in (None, "", "null"):
        return f"experiment={explicit}"

    env_cfg = config.get("environment") or {}
    env_name = env_cfg.get("name")
    algo_cfg = config.get("algorithm") or {}
    algo_target = algo_cfg.get("_target_")
    obs_key = algo_cfg.get("obs_key", "observation")

    for spec in registry:
        env_yaml = REPO_ROOT / "configs" / "environment" / f"{spec.environment_choice}.yaml"
        if not env_yaml.exists():
            continue
        env_choice_name = _read_yaml_scalar(env_yaml, "name")
        if env_choice_name != env_name:
            continue

        algo_yaml = REPO_ROOT / "configs" / "algorithm" / f"{spec.algorithm_choice}.yaml"
        if not algo_yaml.exists():
            continue
        algo_target_expected = _read_yaml_scalar(algo_yaml, "_target_")
        if algo_target_expected != algo_target:
            continue

        if spec.algorithm_choice == "dqn_atari" and obs_key != "pixels":
            continue
        if spec.algorithm_choice == "dqn" and obs_key not in (None, "observation"):
            continue

        return f"experiment={spec.path}"

    return "—"


def _read_yaml_scalar(path: Path, key: str) -> str | None:
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(rf"^{re.escape(key)}:\s*(.+?)\s*$", line)
        if match:
            return match.group(1).strip().strip("'\"")
    return None


def format_int(n: int | None) -> str:
    if n is None:
        return "—"
    return f"{n:,}"


def format_return(value: float | None) -> str:
    if value is None:
        return "—"
    if abs(value) >= 100:
        return f"{value:,.1f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def get_eval_return(run) -> tuple[float | None, str]:
    """Best available return metric and optional note suffix."""
    summary = run.summary
    eval_mean = summary.get("eval/return_mean")
    if eval_mean is not None:
        try:
            if eval_mean == eval_mean:  # skip NaN
                return float(eval_mean), ""
        except (TypeError, ValueError):
            pass

    try:
        history = run.history(keys=["train/episode_reward"], pandas=False)
        values = [
            float(row["train/episode_reward"])
            for row in history
            if row.get("train/episode_reward") is not None
        ]
        if values:
            return max(values), "best train/episode_reward"
    except Exception:
        pass

    return None, ""


def parse_run(run, registry: list[ExperimentSpec]) -> ResultRow | None:
    config = dict(run.config)
    algo_cfg = config.get("algorithm") or {}
    package = algo_package_from_target(algo_cfg.get("_target_"))
    if package is None:
        return None

    trainer_cfg = config.get("trainer") or {}
    env_cfg = config.get("environment") or {}

    eval_return, metric_note = get_eval_return(run)
    notes = (run.notes or "").strip()
    if metric_note:
        notes = f"{notes}; {metric_note}".strip("; ")

    entity = run.entity
    project = run.project
    run_url = f"https://wandb.ai/{entity}/{project}/runs/{run.id}"

    return ResultRow(
        run_name=run.name,
        run_url=run_url,
        environment=env_cfg.get("name") or "—",
        config=infer_experiment_config(config, registry),
        seed=trainer_cfg.get("seed"),
        frames=trainer_cfg.get("total_frames"),
        eval_return=format_return(eval_return),
        notes=notes or "—",
    )


def row_to_markdown(row: ResultRow) -> str:
    run_cell = f"[{row.run_name}]({row.run_url})"
    seed = str(row.seed) if row.seed is not None else "—"
    return (
        f"| {run_cell} | {row.environment} | `{row.config}` | {seed} | "
        f"{format_int(row.frames)} | {row.eval_return} | {row.notes} |"
    )


def build_table(rows: list[ResultRow]) -> str:
    if not rows:
        return (
            f"{TABLE_HEADER}\n"
            f"{TABLE_SEPARATOR}\n"
            f"| — | — | — | — | — | — | "
            f"No finished runs tagged ``template`` yet — see "
            f"[W&B table]({WANDB_TABLE_URL}) |"
        )

    sorted_rows = sorted(rows, key=lambda r: (r.environment, r.config, r.run_name))
    body = "\n".join(row_to_markdown(r) for r in sorted_rows)
    return f"{TABLE_HEADER}\n{TABLE_SEPARATOR}\n{body}"


def replace_results_table(content: str, new_table: str) -> str:
    pattern = re.compile(
        r"(\| Run \| Environment \| Config \| Seed \| Frames \| Eval return \| Notes \|\n"
        r"\|[-| ]+\|\n)"
        r"(?:\|[^\n]+\|\n)*",
        re.MULTILINE,
    )
    if not pattern.search(content):
        raise ValueError("Could not find experimental results table in README.")
    return pattern.sub(new_table + "\n", content, count=1)


def resolve_entity(api, entity: str | None) -> str:
    """Match ``configs/logger/wandb.yaml``: explicit entity, else env, else login default."""
    if entity:
        return entity
    env_entity = os.environ.get("WANDB_ENTITY")
    if env_entity:
        return env_entity
    return api.default_entity


def list_entity_projects(api, entity: str) -> list[str]:
    try:
        return sorted(project.name for project in api.projects(entity))
    except Exception:
        return []


def format_project_not_found_error(
    *,
    entity: str,
    project: str,
    available_projects: list[str],
) -> str:
    lines = [
        f"W&B project not found: {entity}/{project}",
        "",
        "The script defaults to your logged-in entity (or WANDB_ENTITY), not the "
        f"canonical team project {CANONICAL_ENTITY}/{DEFAULT_PROJECT}.",
    ]
    if available_projects:
        lines.append(
            f"Projects visible under '{entity}': {', '.join(available_projects)}"
        )
    else:
        lines.append(f"No projects visible under entity '{entity}'.")
    lines.extend(
        [
            "",
            "Options:",
            f"  • Shared template benchmarks:  --entity {CANONICAL_ENTITY}  "
            f"(requires team access)",
            f"  • Your own runs:               --entity {entity} --project <name>",
            "  • Set default entity:          export WANDB_ENTITY=your-team",
        ]
    )
    return "\n".join(lines)


def fetch_template_runs(
    *,
    entity: str | None,
    project: str,
    tag: str,
    states: tuple[str, ...] = ("finished",),
):
    import wandb

    api = wandb.Api()
    resolved_entity = resolve_entity(api, entity)
    path = f"{resolved_entity}/{project}"
    filters: dict = {"tags": {"$in": [tag]}}
    if len(states) == 1:
        filters["state"] = states[0]
    elif states:
        filters["state"] = {"$in": list(states)}

    try:
        runs = list(api.runs(path, filters=filters, order="-created_at"))
    except Exception as exc:
        message = str(exc)
        if "Could not find project" in message or "404" in message:
            available = list_entity_projects(api, resolved_entity)
            raise RuntimeError(
                format_project_not_found_error(
                    entity=resolved_entity,
                    project=project,
                    available_projects=available,
                )
            ) from exc
        raise

    return runs, resolved_entity


def group_rows_by_algo(
    runs,
    registry: list[ExperimentSpec],
) -> dict[str, list[ResultRow]]:
    grouped: dict[str, list[ResultRow]] = {pkg: [] for pkg in set(ALGO_TARGET_PREFIXES.values())}
    for run in runs:
        row = parse_run(run, registry)
        if row is None:
            continue
        config = dict(run.config)
        algo_cfg = config.get("algorithm") or {}
        package = algo_package_from_target(algo_cfg.get("_target_"))
        if package is not None:
            grouped[package].append(row)
    return grouped


def update_readme(path: Path, rows: list[ResultRow], *, dry_run: bool) -> bool:
    content = path.read_text(encoding="utf-8")
    new_table = build_table(rows)
    updated = replace_results_table(content, new_table)
    if updated == content:
        return False
    if not dry_run:
        path.write_text(updated, encoding="utf-8")
    return True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Populate algorithm README experimental-results tables from W&B.",
    )
    parser.add_argument(
        "--entity",
        default=None,
        help=(
            "W&B entity (team/user). Defaults to WANDB_ENTITY or the logged-in "
            f"user's default entity. Canonical shared benchmarks live under "
            f"{CANONICAL_ENTITY}."
        ),
    )
    parser.add_argument(
        "--project",
        default=DEFAULT_PROJECT,
        help="W&B project name.",
    )
    parser.add_argument(
        "--tag",
        default="template",
        help="Only include runs with this W&B tag (default: template).",
    )
    parser.add_argument(
        "--algo",
        choices=sorted(set(ALGO_TARGET_PREFIXES.values())),
        nargs="*",
        help="Limit updates to these algorithm packages (default: all).",
    )
    parser.add_argument(
        "--include-running",
        action="store_true",
        help="Also include runs that are still running (default: finished only).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print tables without modifying README files.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    registry = load_experiment_registry()

    states: tuple[str, ...]
    if args.include_running:
        states = ("finished", "running")
    else:
        states = ("finished",)

    try:
        runs, resolved_entity = fetch_template_runs(
            entity=args.entity,
            project=args.project,
            tag=args.tag,
            states=states,
        )
    except Exception as exc:
        print(f"Failed to fetch W&B runs:\n{exc}", file=sys.stderr)
        if "W&B project not found" not in str(exc):
            print(
                "Authenticate with `wandb login` or set WANDB_API_KEY, then retry.",
                file=sys.stderr,
            )
        return 1

    grouped = group_rows_by_algo(runs, registry)
    targets = args.algo or sorted(grouped.keys())

    print(
        f"Fetched {len(runs)} run(s) tagged '{args.tag}' "
        f"from {resolved_entity}/{args.project}."
    )

    changed = 0
    for algo in targets:
        readme = algo_readme_path(algo)
        if not readme.exists():
            print(f"Skipping missing README: {readme}", file=sys.stderr)
            continue
        rows = grouped.get(algo, [])
        print(f"\n## {algo} ({len(rows)} run(s))")
        table = build_table(rows)
        print(table)
        if update_readme(readme, rows, dry_run=args.dry_run):
            changed += 1
            action = "Would update" if args.dry_run else "Updated"
            print(f"{action}: {readme.relative_to(REPO_ROOT)}")

    if args.dry_run:
        print(f"\nDry run complete — {changed} README(s) would change.")
    else:
        print(f"\nDone — {changed} README(s) updated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
