from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

PROCESSED = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

OUTPUT_FILE = (
    PROCESSED
    / "v3_promotion_gate.json"
)

MIN_TRADES = 50
MIN_TRADES_PER_DAY = 4.0
MIN_WIN_RATE_PCT = 80.0
MAX_MEDIAN_GAP_HOURS = 2.0
MIN_CONFIDENCE = 0.80


def matching_strict_pairs(
    frame: pd.DataFrame,
    key_columns: list[str],
    source: str,
) -> list[dict]:
    if frame.empty:
        return []

    validation = frame[
        (
            frame[
                "period"
            ].astype(str)
            == "validation"
        )
        & (
            frame[
                "strict_target_met"
            ].astype(bool)
        )
    ].copy()

    test = frame[
        (
            frame[
                "period"
            ].astype(str)
            == "test"
        )
        & (
            frame[
                "strict_target_met"
            ].astype(bool)
        )
    ].copy()

    if (
        validation.empty
        or test.empty
    ):
        return []

    candidates = []

    for _, val in validation.iterrows():
        mask = pd.Series(
            True,
            index=test.index,
        )

        for column in key_columns:
            if (
                pd.api.types
                .is_numeric_dtype(
                    frame[column]
                )
            ):
                mask &= np.isclose(
                    pd.to_numeric(
                        test[column],
                        errors="coerce",
                    ),
                    float(
                        val[column]
                    ),
                    equal_nan=False,
                )
            else:
                mask &= (
                    test[column]
                    .astype(str)
                    == str(
                        val[column]
                    )
                )

        matches = test[
            mask
        ]

        for _, later in (
            matches.iterrows()
        ):
            candidates.append(
                {
                    "source":
                        source,
                    "configuration": {
                        column: (
                            float(
                                val[column]
                            )
                            if isinstance(
                                val[column],
                                (
                                    float,
                                    np.floating,
                                ),
                            )
                            else int(
                                val[column]
                            )
                            if isinstance(
                                val[column],
                                (
                                    int,
                                    np.integer,
                                ),
                            )
                            else str(
                                val[column]
                            )
                        )
                        for column
                        in key_columns
                    },
                    "validation": {
                        "trades":
                            int(
                                val[
                                    "trades"
                                ]
                            ),
                        "trades_per_day":
                            float(
                                val[
                                    "trades_per_day"
                                ]
                            ),
                        "median_gap_hours":
                            float(
                                val[
                                    "median_gap_hours"
                                ]
                            ),
                        "win_rate_mid_pct":
                            float(
                                val[
                                    "win_rate_mid_pct"
                                ]
                            ),
                        "net_mid_sum_pct":
                            float(
                                val[
                                    "net_mid_sum_pct"
                                ]
                            ),
                        "net_high_sum_pct":
                            float(
                                val[
                                    "net_high_sum_pct"
                                ]
                            ),
                    },
                    "test": {
                        "trades":
                            int(
                                later[
                                    "trades"
                                ]
                            ),
                        "trades_per_day":
                            float(
                                later[
                                    "trades_per_day"
                                ]
                            ),
                        "median_gap_hours":
                            float(
                                later[
                                    "median_gap_hours"
                                ]
                            ),
                        "win_rate_mid_pct":
                            float(
                                later[
                                    "win_rate_mid_pct"
                                ]
                            ),
                        "net_mid_sum_pct":
                            float(
                                later[
                                    "net_mid_sum_pct"
                                ]
                            ),
                        "net_high_sum_pct":
                            float(
                                later[
                                    "net_high_sum_pct"
                                ]
                            ),
                    },
                }
            )

    return candidates


def load_csv(
    path: Path,
) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(
            path
        )
    except (
        OSError,
        ValueError,
    ):
        return pd.DataFrame()


def main() -> None:
    print(
        "TRADEMIND V3 - STRICT "
        "CANDIDATE PROMOTION GATE"
    )

    print()
    print(
        "A candidate is never promoted "
        "because of confidence alone."
    )

    print(
        "Both validation and later test "
        "must independently satisfy:"
    )

    print(
        f"  confidence >= "
        f"{MIN_CONFIDENCE:.0%}"
    )

    print(
        f"  win rate >= "
        f"{MIN_WIN_RATE_PCT:.1f}%"
    )

    print(
        f"  trades/day >= "
        f"{MIN_TRADES_PER_DAY:.1f}"
    )

    print(
        f"  median gap <= "
        f"{MAX_MEDIAN_GAP_HOURS:.1f}h"
    )

    print(
        f"  trades >= "
        f"{MIN_TRADES}"
    )

    print(
        "  positive mid/high-cost net"
    )

    candidates = []

    multiframe = load_csv(
        PROCESSED
        / "v3_multitimeframe_profit_first"
        / "candidate_results.csv"
    )

    if not multiframe.empty:
        candidates.extend(
            matching_strict_pairs(
                multiframe,
                [
                    "timeframe",
                    "horizon_minutes",
                    "proposer_threshold",
                    "selector_threshold",
                ],
                "v3_multitimeframe_profit_first",
            )
        )

    barrier = load_csv(
        PROCESSED
        / "v3_barrier_opportunity"
        / "candidate_results.csv"
    )

    if not barrier.empty:
        candidates.extend(
            matching_strict_pairs(
                barrier,
                [
                    "horizon_minutes",
                    "take_profit_pct",
                    "stop_loss_pct",
                    "confidence_threshold",
                ],
                "v3_barrier_opportunity",
            )
        )

    transformer = load_csv(
        PROCESSED
        / "v3_transformer_barrier"
        / "threshold_results.csv"
    )

    if not transformer.empty:
        candidates.extend(
            matching_strict_pairs(
                transformer,
                [
                    "horizon_minutes",
                    "take_profit_pct",
                    "stop_loss_pct",
                    "confidence_threshold",
                ],
                "v3_transformer_barrier",
            )
        )

    if candidates:
        candidates.sort(
            key=lambda item: (
                item[
                    "test"
                ][
                    "win_rate_mid_pct"
                ],
                item[
                    "test"
                ][
                    "trades_per_day"
                ],
                item[
                    "test"
                ][
                    "net_high_sum_pct"
                ],
            ),
            reverse=True,
        )

        best = candidates[0]

        status = (
            "QUALIFIED_FOR_FORWARD_PAPER"
        )

        note = (
            "A research candidate satisfied "
            "the strict validation and later-test "
            "gate. It is eligible only for a new "
            "frozen forward-paper candidate. "
            "It is not approved for real-money "
            "trading."
        )
    else:
        best = None

        status = (
            "NO_QUALIFIED_CANDIDATE"
        )

        note = (
            "No V3 experiment satisfied the "
            "required confidence, win rate, "
            "frequency, sample size and cost "
            "robustness on both validation and "
            "later test. Existing V1/V2 remain "
            "unchanged."
        )

    payload = {
        "research_only":
            True,
        "status":
            status,
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
        "qualified_candidates":
            candidates,
        "selected_candidate":
            best,
        "note":
            note,
    }

    OUTPUT_FILE.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        f"Status: {status}"
    )

    print(
        f"Qualified candidates: "
        f"{len(candidates)}"
    )

    if best is not None:
        print()
        print(
            "Selected candidate:"
        )

        print(
            json.dumps(
                best,
                indent=2,
            )
        )

    print()
    print(note)

    print()
    print(
        f"Saved: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
