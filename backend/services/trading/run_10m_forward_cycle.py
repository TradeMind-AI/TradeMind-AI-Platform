from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]


STEPS = [
    (
        "1. Refresh closed BTC 1m candles",
        PROJECT_ROOT
        / "backend"
        / "services"
        / "data_pipeline"
        / "update_live_1m_data.py",
        [],
    ),
    (
        "2. Rebuild complete 10m candles",
        PROJECT_ROOT
        / "backend"
        / "services"
        / "data_pipeline"
        / "resample_timeframes.py",
        [
            "--timeframes",
            "10m",
        ],
    ),
    (
        "3. Build live 10m features",
        PROJECT_ROOT
        / "backend"
        / "services"
        / "data_pipeline"
        / "build_live_features_10m.py",
        [],
    ),
    (
        "4. Update frozen 10m forward paper log",
        PROJECT_ROOT
        / "backend"
        / "services"
        / "trading"
        / "paper_forward_10m.py",
        [],
    ),
]


def run_step(
    label: str,
    script: Path,
    args: list[str],
) -> None:
    print()
    print("=" * 78)
    print(label)
    print("=" * 78)

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            *args,
        ],
        cwd=PROJECT_ROOT,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"{label} failed with "
            f"exit code "
            f"{result.returncode}."
        )


def main() -> None:
    print(
        "TRADEMIND AI - "
        "COST_AWARE_10M_V1 "
        "FORWARD PAPER CYCLE"
    )

    if not (
        PROJECT_ROOT
        / "backend"
        / "models"
        / "locked_xgboost_10m_bundle.joblib"
    ).exists():
        raise FileNotFoundError(
            "10m frozen bundle does not "
            "exist. Run "
            "train_cost_aware_10m_paper_model.py "
            "once first."
        )

    for label, script, args in STEPS:
        run_step(
            label,
            script,
            args,
        )

    print()
    print("=" * 78)
    print(
        "10M FORWARD PAPER CYCLE COMPLETE"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
