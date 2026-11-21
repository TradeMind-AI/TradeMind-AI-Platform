from __future__ import annotations

from pathlib import Path
import json


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = PROCESSED / "v4_promotion_gate.json"

MIN_TRADES = 50
MIN_TRADES_PER_DAY = 4.0
MIN_WIN_RATE_PCT = 80.0
MAX_MEDIAN_GAP_HOURS = 2.0
MIN_CONFIDENCE = 0.80

SOURCES = [
    ("v4_xgb_tuning", "summary.json"),
    ("v4_precision_tuning", "summary.json"),
    ("v4_transformer_tuning", "summary.json"),
    ("v4_regime_tuning", "summary.json"),
]


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except (
        OSError,
        json.JSONDecodeError,
    ):
        return {}


def normalize_metrics(row: dict) -> dict:
    return {
        "trades": int(
            row.get("trades", 0)
        ),
        "trades_per_day": float(
            row.get(
                "trades_per_day",
                0.0,
            )
        ),
        "median_gap_hours": float(
            row.get(
                "median_gap_hours",
                float("inf"),
            )
        ),
        "win_rate_mid_pct": float(
            row.get(
                "win_rate_mid_pct",
                0.0,
            )
        ),
        "net_mid_sum_pct": float(
            row.get(
                "net_mid_sum_pct",
                0.0,
            )
        ),
        "net_high_sum_pct": float(
            row.get(
                "net_high_sum_pct",
                0.0,
            )
        ),
    }


def threshold_from(row: dict) -> float:
    for key in (
        "confidence_threshold",
        "threshold",
        "selector_threshold",
    ):
        if key in row:
            try:
                return float(row[key])
            except (
                TypeError,
                ValueError,
            ):
                pass
    return 0.0


def strict(
    row: dict,
) -> bool:
    metrics = normalize_metrics(row)
    threshold = threshold_from(row)

    return bool(
        threshold >= MIN_CONFIDENCE
        and metrics["trades"]
        >= MIN_TRADES
        and metrics["trades_per_day"]
        >= MIN_TRADES_PER_DAY
        and metrics["median_gap_hours"]
        <= MAX_MEDIAN_GAP_HOURS
        and metrics["win_rate_mid_pct"]
        >= MIN_WIN_RATE_PCT
        and metrics["net_mid_sum_pct"] > 0.0
        and metrics["net_high_sum_pct"] > 0.0
    )


def main() -> None:
    tracks = []
    qualified = []

    for directory, filename in SOURCES:
        summary = load_json(
            PROCESSED
            / directory
            / filename
        )

        if not summary:
            tracks.append(
                {
                    "source": directory,
                    "available": False,
                }
            )
            continue

        validation = (
            summary.get(
                "selected_validation"
            )
            or summary.get(
                "validation_selected"
            )
            or {}
        )

        later_test = (
            summary.get(
                "later_test"
            )
            or summary.get(
                "test_result"
            )
            or {}
        )

        validation_pass = strict(
            validation
        )
        test_pass = strict(
            later_test
        )

        item = {
            "source": directory,
            "available": True,
            "validation_pass":
                validation_pass,
            "later_test_pass":
                test_pass,
            "validation":
                normalize_metrics(
                    validation
                ),
            "later_test":
                normalize_metrics(
                    later_test
                ),
            "validation_threshold":
                threshold_from(
                    validation
                ),
            "test_threshold":
                threshold_from(
                    later_test
                ),
        }

        tracks.append(item)

        if (
            validation_pass
            and test_pass
        ):
            qualified.append(
                item
            )

    status = (
        "QUALIFIED_FOR_FORWARD_PAPER"
        if qualified
        else "NO_QUALIFIED_CANDIDATE"
    )

    payload = {
        "research_only": True,
        "status": status,
        "requirements": {
            "minimum_confidence":
                MIN_CONFIDENCE,
            "minimum_win_rate_pct":
                MIN_WIN_RATE_PCT,
            "minimum_trades_per_day":
                MIN_TRADES_PER_DAY,
            "maximum_median_gap_hours":
                MAX_MEDIAN_GAP_HOURS,
            "minimum_trades":
                MIN_TRADES,
            "positive_mid_cost_net":
                True,
            "positive_high_cost_net":
                True,
        },
        "tracks": tracks,
        "qualified_candidates":
            qualified,
        "selected_candidate":
            (
                qualified[0]
                if qualified
                else None
            ),
        "note": (
            "No V4 model is promoted unless both "
            "validation and the untouched later test "
            "independently satisfy confidence, win-rate, "
            "frequency, sample-size, and transaction-cost "
            "requirements."
        ),
    }

    OUTPUT_FILE.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "TRADEMIND V4 PROMOTION GATE"
    )
    print(
        f"Status: {status}"
    )

    for item in tracks:
        print()
        print(
            f"{item['source']}:"
        )
        if not item["available"]:
            print("  not available")
            continue
        print(
            "  validation pass: "
            f"{item['validation_pass']}"
        )
        print(
            "  later test pass: "
            f"{item['later_test_pass']}"
        )
        print(
            "  later test trades/day: "
            f"{item['later_test']['trades_per_day']:.3f}"
        )
        print(
            "  later test win rate: "
            f"{item['later_test']['win_rate_mid_pct']:.2f}%"
        )
        print(
            "  later test high-cost net: "
            f"{item['later_test']['net_high_sum_pct']:.2f}%"
        )

    print()
    print(
        f"Qualified candidates: "
        f"{len(qualified)}"
    )
    print(
        f"Saved: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
