from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]


STEPS = [
    (
        "1. Update live market data",
        PROJECT_ROOT
        / "backend"
        / "services"
        / "data_pipeline"
        / "update_live_market_data.py",
    ),
    (
        "2. Clean BTC data",
        PROJECT_ROOT
        / "backend"
        / "services"
        / "data_pipeline"
        / "clean_btc_data.py",
    ),
    (
        "3. Build live ML features",
        PROJECT_ROOT
        / "backend"
        / "services"
        / "data_pipeline"
        / "build_live_features.py",
    ),
    (
        "4. Settle previous paper trades",
        PROJECT_ROOT
        / "backend"
        / "services"
        / "trading"
        / "settle_paper_trades.py",
    ),
    (
        "5. Generate new paper signal",
        PROJECT_ROOT
        / "backend"
        / "services"
        / "trading"
        / "paper_trader.py",
    ),
    (
        "6. Generate paper trading report",
        PROJECT_ROOT
        / "backend"
        / "services"
        / "trading"
        / "paper_report.py",
    ),
]


def run_step(name: str, script: Path):
    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)

    if not script.exists():
        raise FileNotFoundError(
            f"Required script not found:\n{script}"
        )

    result = subprocess.run(
        [
            sys.executable,
            str(script),
        ],
        cwd=PROJECT_ROOT,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"{name} failed with "
            f"exit code {result.returncode}."
        )

    print(
        f"\n{name}: COMPLETE"
    )


def main():
    print(
        "=" * 80
    )

    print(
        "TRADEMIND AI - PAPER TRADING CYCLE"
    )

    print(
        "=" * 80
    )

    print(
        f"Project: {PROJECT_ROOT}"
    )

    try:
        for name, script in STEPS:
            run_step(
                name,
                script,
            )

    except Exception as error:
        print(
            "\n"
            + "=" * 80
        )

        print(
            "PAPER TRADING CYCLE FAILED"
        )

        print(
            "=" * 80
        )

        print(
            f"\n{error}"
        )

        sys.exit(1)

    print(
        "\n"
        + "=" * 80
    )

    print(
        "PAPER TRADING CYCLE COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        "\nCompleted:"
    )

    print(
        "Live BTC/ETH/BNB/funding refreshed"
    )

    print(
        "BTC cleaned"
    )

    print(
        "84 live ML features rebuilt"
    )

    print(
        "Previous trades settled"
    )

    print(
        "Latest paper signal generated"
    )

    print(
        "Paper trading report generated"
    )


if __name__ == "__main__":
    main()