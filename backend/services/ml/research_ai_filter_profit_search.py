from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[3]

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "btcusdt_10m_features.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ai_filter_profit_search"
)

RESULT_FILE = OUTPUT_DIR / "search_results.csv"
TRADE_FILE = OUTPUT_DIR / "search_trades.csv"
TOP_FILE = OUTPUT_DIR / "top_candidates.csv"
SUMMARY_FILE = OUTPUT_DIR / "summary.json"

RESEARCH_START = pd.Timestamp(
    "2023-03-26",
    tz="UTC",
)

VALIDATION_START = pd.Timestamp(
    "2025-01-01",
    tz="UTC",
)

TEST_START = pd.Timestamp(
    "2025-10-01",
    tz="UTC",
)

HORIZON_BARS = 6
TARGET_THRESHOLD = 0.0040

COSTS = {
    "low": 0.0006,
    "mid": 0.0014,
    "high": 0.0027,
}

CONFIDENCE_THRESHOLDS = [
    0.55,
    0.60,
    0.65,
]

RSI_RULES = [
    None,
    (25.0, 75.0),
    (30.0, 70.0),
    (35.0, 65.0),
    (40.0, 60.0),
]

BOOLEAN_OPTIONS = [
    False,
    True,
]

RELATIVE_VOLUME_THRESHOLDS = [
    None,
    1.0,
    1.25,
]

MIN_TOTAL_TRADES = 15

NON_FEATURE_COLUMNS = {
    "open_time",
    "close_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
    "future_return_10m",
    "target",
}


@dataclass(frozen=True)
class Rule:
    confidence: float
    rsi_rule: tuple[float, float] | None
    ema_alignment: bool
    macd_alignment: bool
    bollinger_reversion: bool
    relative_volume_min: float | None

    @property
    def key(self) -> str:
        parts = [
            f"conf{self.confidence:.2f}",
        ]

        if self.rsi_rule is not None:
            parts.append(
                "rsi"
                + str(
                    int(
                        self.rsi_rule[0]
                    )
                )
                + "_"
                + str(
                    int(
                        self.rsi_rule[1]
                    )
                )
            )

        if self.ema_alignment:
            parts.append("ema")

        if self.macd_alignment:
            parts.append("macd")

        if self.bollinger_reversion:
            parts.append("bb")

        if (
            self.relative_volume_min
            is not None
        ):
            parts.append(
                "rv"
                + str(
                    self.relative_volume_min
                ).replace(
                    ".",
                    "_",
                )
            )

        if len(parts) == 1:
            parts.append("baseline")

        return "__".join(parts)

    @property
    def label(self) -> str:
        parts = [
            f"AI conf≥{self.confidence:.2f}",
        ]

        if self.rsi_rule is not None:
            parts.append(
                "RSI "
                f"{int(self.rsi_rule[0])}/"
                f"{int(self.rsi_rule[1])}"
            )

        if self.ema_alignment:
            parts.append("EMA")

        if self.macd_alignment:
            parts.append("MACD")

        if self.bollinger_reversion:
            parts.append("Bollinger")

        if (
            self.relative_volume_min
            is not None
        ):
            parts.append(
                "RelVol≥"
                f"{self.relative_volume_min:.2f}x"
            )

        return " + ".join(parts)


def make_target(
    returns: pd.Series,
) -> pd.Series:
    return pd.Series(
        np.select(
            [
                returns < -TARGET_THRESHOLD,
                returns > TARGET_THRESHOLD,
            ],
            [
                "DOWN",
                "UP",
            ],
            default="NEUTRAL",
        ),
        index=returns.index,
    )


def ema(
    series: pd.Series,
    span: int,
) -> pd.Series:
    return series.ewm(
        span=span,
        adjust=False,
        min_periods=span,
    ).mean()


def add_indicators(
    df: pd.DataFrame,
) -> pd.DataFrame:
    result = df.copy()

    close = result["close"]

    result["ema20_filter"] = ema(
        close,
        20,
    )

    delta = close.diff()

    gain = delta.clip(
        lower=0.0,
    )

    loss = (
        -delta.clip(
            upper=0.0,
        )
    )

    average_gain = gain.ewm(
        alpha=1.0 / 14.0,
        adjust=False,
        min_periods=14,
    ).mean()

    average_loss = loss.ewm(
        alpha=1.0 / 14.0,
        adjust=False,
        min_periods=14,
    ).mean()

    rs = (
        average_gain
        / average_loss.replace(
            0.0,
            np.nan,
        )
    )

    result["rsi14_filter"] = (
        100.0
        - 100.0
        / (
            1.0
            + rs
        )
    )

    sma20 = close.rolling(
        20,
        min_periods=20,
    ).mean()

    std20 = close.rolling(
        20,
        min_periods=20,
    ).std(
        ddof=0,
    )

    result["bb_upper_filter"] = (
        sma20
        + 2.0 * std20
    )

    result["bb_lower_filter"] = (
        sma20
        - 2.0 * std20
    )

    ema12 = ema(
        close,
        12,
    )

    ema26 = ema(
        close,
        26,
    )

    macd = (
        ema12
        - ema26
    )

    macd_signal = macd.ewm(
        span=9,
        adjust=False,
        min_periods=9,
    ).mean()

    result["macd_hist_filter"] = (
        macd
        - macd_signal
    )

    volume_mean20 = (
        result["volume"]
        .rolling(
            20,
            min_periods=20,
        )
        .mean()
    )

    result["relative_volume_filter"] = (
        result["volume"]
        / volume_mean20.replace(
            0.0,
            np.nan,
        )
    )

    return result


def train_model(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[
    XGBClassifier,
    LabelEncoder,
]:
    train = df[
        df["open_time"]
        < VALIDATION_START
    ].copy()

    future_return = (
        train["close"].shift(
            -HORIZON_BARS
        )
        / train["close"]
        - 1.0
    )

    mask = future_return.notna()

    X = train.loc[
        mask,
        feature_columns,
    ]

    y_text = make_target(
        future_return.loc[
            mask
        ]
    )

    encoder = LabelEncoder()

    encoder.fit(
        [
            "DOWN",
            "NEUTRAL",
            "UP",
        ]
    )

    y = encoder.transform(
        y_text
    )

    model = XGBClassifier(
        n_estimators=250,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.0,
        reg_lambda=1.0,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )

    model.fit(
        X,
        y,
    )

    return (
        model,
        encoder,
    )


def add_predictions(
    df: pd.DataFrame,
    feature_columns: list[str],
    model: XGBClassifier,
    encoder: LabelEncoder,
) -> pd.DataFrame:
    result = df.copy()

    mask = (
        result["open_time"]
        >= VALIDATION_START
    )

    X = result.loc[
        mask,
        feature_columns,
    ]

    probabilities = (
        model.predict_proba(
            X
        )
    )

    encoded = model.predict(
        X
    )

    labels = (
        encoder.inverse_transform(
            encoded
        )
    )

    result["ai_prediction"] = pd.NA
    result["ai_confidence"] = np.nan

    result.loc[
        mask,
        "ai_prediction",
    ] = labels

    result.loc[
        mask,
        "ai_confidence",
    ] = probabilities.max(
        axis=1
    )

    return result


def rule_allows(
    row: pd.Series,
    direction: int,
    rule: Rule,
) -> bool:
    if (
        float(
            row["ai_confidence"]
        )
        < rule.confidence
    ):
        return False

    if rule.rsi_rule is not None:
        rsi_value = row[
            "rsi14_filter"
        ]

        if pd.isna(rsi_value):
            return False

        long_max, short_min = (
            rule.rsi_rule
        )

        if (
            direction == 1
            and float(
                rsi_value
            ) > long_max
        ):
            return False

        if (
            direction == -1
            and float(
                rsi_value
            ) < short_min
        ):
            return False

    if rule.ema_alignment:
        ema20 = row[
            "ema20_filter"
        ]

        if pd.isna(ema20):
            return False

        close = float(
            row["close"]
        )

        if (
            direction == 1
            and close <= float(
                ema20
            )
        ):
            return False

        if (
            direction == -1
            and close >= float(
                ema20
            )
        ):
            return False

    if rule.macd_alignment:
        macd_hist = row[
            "macd_hist_filter"
        ]

        if pd.isna(macd_hist):
            return False

        if (
            direction == 1
            and float(
                macd_hist
            ) <= 0.0
        ):
            return False

        if (
            direction == -1
            and float(
                macd_hist
            ) >= 0.0
        ):
            return False

    if rule.bollinger_reversion:
        upper = row[
            "bb_upper_filter"
        ]

        lower = row[
            "bb_lower_filter"
        ]

        if (
            pd.isna(upper)
            or pd.isna(lower)
        ):
            return False

        close = float(
            row["close"]
        )

        if (
            direction == 1
            and close > float(
                lower
            )
        ):
            return False

        if (
            direction == -1
            and close < float(
                upper
            )
        ):
            return False

    if (
        rule.relative_volume_min
        is not None
    ):
        relative_volume = row[
            "relative_volume_filter"
        ]

        if (
            pd.isna(
                relative_volume
            )
            or float(
                relative_volume
            )
            < rule.relative_volume_min
        ):
            return False

    return True


def simulate_rule(
    df: pd.DataFrame,
    rule: Rule,
) -> list[dict]:
    trades = []

    index = 0

    while (
        index
        < len(df)
        - HORIZON_BARS
        - 1
    ):
        row = df.iloc[index]

        signal_time = row[
            "open_time"
        ]

        if signal_time < VALIDATION_START:
            index += 1
            continue

        prediction = row[
            "ai_prediction"
        ]

        confidence = row[
            "ai_confidence"
        ]

        if (
            pd.isna(prediction)
            or pd.isna(confidence)
            or prediction
            not in {
                "UP",
                "DOWN",
            }
        ):
            index += 1
            continue

        direction = (
            1
            if prediction == "UP"
            else -1
        )

        if not rule_allows(
            row,
            direction,
            rule,
        ):
            index += 1
            continue

        entry_index = (
            index + 1
        )

        exit_index = (
            index
            + HORIZON_BARS
        )

        if exit_index >= len(df):
            break

        entry_row = (
            df.iloc[
                entry_index
            ]
        )

        exit_row = (
            df.iloc[
                exit_index
            ]
        )

        entry_price = float(
            entry_row[
                "open"
            ]
        )

        exit_price = float(
            exit_row[
                "close"
            ]
        )

        gross_return = (
            (
                exit_price
                / entry_price
                - 1.0
            )
            * direction
        )

        period = (
            "validation"
            if signal_time
            < TEST_START
            else "test"
        )

        trade = {
            "rule_key":
                rule.key,
            "rule":
                rule.label,
            "period":
                period,
            "signal_time":
                signal_time,
            "direction":
                (
                    "LONG"
                    if direction == 1
                    else "SHORT"
                ),
            "confidence":
                float(
                    confidence
                ),
            "rsi14":
                row[
                    "rsi14_filter"
                ],
            "relative_volume":
                row[
                    "relative_volume_filter"
                ],
            "entry_price":
                entry_price,
            "exit_price":
                exit_price,
            "gross_return_pct":
                gross_return
                * 100.0,
        }

        for (
            cost_name,
            cost,
        ) in COSTS.items():
            trade[
                f"net_{cost_name}_pct"
            ] = (
                gross_return
                - cost
            ) * 100.0

        trades.append(
            trade
        )

        index = exit_index + 1

    return trades


def max_drawdown_from_returns(
    values: np.ndarray,
) -> float:
    if len(values) == 0:
        return 0.0

    equity = np.cumprod(
        1.0
        + values / 100.0
    )

    peaks = np.maximum.accumulate(
        equity
    )

    drawdowns = (
        equity / peaks
        - 1.0
    )

    return float(
        drawdowns.min()
        * 100.0
    )


def summarize_rule(
    trades: pd.DataFrame,
    rule: Rule,
) -> dict:
    subset = trades[
        trades["rule_key"]
        == rule.key
    ].copy()

    validation = subset[
        subset["period"]
        == "validation"
    ]

    test = subset[
        subset["period"]
        == "test"
    ]

    def metrics(
        frame: pd.DataFrame,
    ) -> dict:
        if frame.empty:
            return {
                "trades": 0,
                "win_rate_mid_pct": 0.0,
                "avg_net_mid_pct": 0.0,
                "net_mid_sum_pct": 0.0,
                "net_high_sum_pct": 0.0,
                "max_drawdown_mid_pct": 0.0,
            }

        mid = frame[
            "net_mid_pct"
        ].to_numpy(
            dtype=float
        )

        return {
            "trades":
                int(
                    len(
                        frame
                    )
                ),
            "win_rate_mid_pct":
                float(
                    (
                        mid > 0.0
                    ).mean()
                    * 100.0
                ),
            "avg_net_mid_pct":
                float(
                    mid.mean()
                ),
            "net_mid_sum_pct":
                float(
                    mid.sum()
                ),
            "net_high_sum_pct":
                float(
                    frame[
                        "net_high_pct"
                    ].sum()
                ),
            "max_drawdown_mid_pct":
                max_drawdown_from_returns(
                    mid
                ),
        }

    validation_metrics = metrics(
        validation
    )

    test_metrics = metrics(
        test
    )

    combined = metrics(
        subset
    )

    total_trades = combined[
        "trades"
    ]

    score = (
        combined[
            "avg_net_mid_pct"
        ]
        * np.log1p(
            total_trades
        )
        + 0.01
        * combined[
            "win_rate_mid_pct"
        ]
        + 0.002
        * combined[
            "net_high_sum_pct"
        ]
    )

    return {
        "rule_key":
            rule.key,
        "rule":
            rule.label,
        "confidence_threshold":
            rule.confidence,
        "rsi_rule":
            (
                ""
                if rule.rsi_rule
                is None
                else (
                    f"{int(rule.rsi_rule[0])}/"
                    f"{int(rule.rsi_rule[1])}"
                )
            ),
        "ema_alignment":
            int(
                rule.ema_alignment
            ),
        "macd_alignment":
            int(
                rule.macd_alignment
            ),
        "bollinger_reversion":
            int(
                rule.bollinger_reversion
            ),
        "relative_volume_min":
            (
                np.nan
                if rule.relative_volume_min
                is None
                else rule.relative_volume_min
            ),
        "validation_trades":
            validation_metrics[
                "trades"
            ],
        "validation_win_rate_mid_pct":
            validation_metrics[
                "win_rate_mid_pct"
            ],
        "validation_avg_net_mid_pct":
            validation_metrics[
                "avg_net_mid_pct"
            ],
        "validation_net_mid_sum_pct":
            validation_metrics[
                "net_mid_sum_pct"
            ],
        "test_trades":
            test_metrics[
                "trades"
            ],
        "test_win_rate_mid_pct":
            test_metrics[
                "win_rate_mid_pct"
            ],
        "test_avg_net_mid_pct":
            test_metrics[
                "avg_net_mid_pct"
            ],
        "test_net_mid_sum_pct":
            test_metrics[
                "net_mid_sum_pct"
            ],
        "total_trades":
            total_trades,
        "combined_win_rate_mid_pct":
            combined[
                "win_rate_mid_pct"
            ],
        "combined_avg_net_mid_pct":
            combined[
                "avg_net_mid_pct"
            ],
        "combined_net_mid_sum_pct":
            combined[
                "net_mid_sum_pct"
            ],
        "combined_net_high_sum_pct":
            combined[
                "net_high_sum_pct"
            ],
        "combined_max_drawdown_mid_pct":
            combined[
                "max_drawdown_mid_pct"
            ],
        "exploratory_score":
            float(
                score
            ),
    }


def build_rules() -> list[Rule]:
    rules = []

    for (
        confidence,
        rsi_rule,
        ema_alignment,
        macd_alignment,
        bollinger_reversion,
        relative_volume_min,
    ) in product(
        CONFIDENCE_THRESHOLDS,
        RSI_RULES,
        BOOLEAN_OPTIONS,
        BOOLEAN_OPTIONS,
        BOOLEAN_OPTIONS,
        RELATIVE_VOLUME_THRESHOLDS,
    ):
        # EMA alignment and Bollinger mean-reversion often
        # express opposite regimes. Keep the combination
        # available for exploration, but avoid duplicate
        # pure-baseline forms.
        rule = Rule(
            confidence=
                confidence,
            rsi_rule=
                rsi_rule,
            ema_alignment=
                ema_alignment,
            macd_alignment=
                macd_alignment,
            bollinger_reversion=
                bollinger_reversion,
            relative_volume_min=
                relative_volume_min,
        )

        rules.append(
            rule
        )

    unique = {}

    for rule in rules:
        unique[
            rule.key
        ] = rule

    return list(
        unique.values()
    )


def main() -> None:
    print(
        "TRADEMIND - EXPLORATORY AI FILTER PROFIT SEARCH"
    )
    print()

    print(
        "WARNING: validation and test are both used for exploration here."
    )

    print(
        "Any winner must be treated as a NEW candidate and forward-tested."
    )
    print()

    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Missing feature file: {FEATURE_FILE}"
        )

    df = pd.read_csv(
        FEATURE_FILE
    )

    df["open_time"] = pd.to_datetime(
        df["open_time"],
        utc=True,
        errors="coerce",
    )

    required = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    for column in required:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = (
        df.dropna(
            subset=[
                "open_time",
                *required,
            ]
        )
        .sort_values(
            "open_time"
        )
        .drop_duplicates(
            subset=["open_time"],
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )

    df = df[
        df["open_time"]
        >= RESEARCH_START
    ].copy().reset_index(
        drop=True
    )

    df = add_indicators(
        df
    )

    filter_columns = {
        "ema20_filter",
        "rsi14_filter",
        "bb_upper_filter",
        "bb_lower_filter",
        "macd_hist_filter",
        "relative_volume_filter",
    }

    feature_columns = [
        column
        for column in df.columns
        if (
            column
            not in NON_FEATURE_COLUMNS
            and column
            not in filter_columns
            and pd.api.types.is_numeric_dtype(
                df[column]
            )
        )
    ]

    model_rows = df[
        [
            "open_time",
            *feature_columns,
        ]
    ].dropna()

    valid_times = set(
        model_rows[
            "open_time"
        ]
    )

    df = df[
        df["open_time"]
        .isin(
            valid_times
        )
    ].copy().reset_index(
        drop=True
    )

    model, encoder = train_model(
        df,
        feature_columns,
    )

    df = add_predictions(
        df,
        feature_columns,
        model,
        encoder,
    )

    rules = build_rules()

    print(
        f"Candidate rules: {len(rules):,}"
    )

    all_trades = []
    result_rows = []

    for index, rule in enumerate(
        rules,
        start=1,
    ):
        rule_trades = (
            simulate_rule(
                df,
                rule,
            )
        )

        if rule_trades:
            frame = pd.DataFrame(
                rule_trades
            )

            all_trades.append(
                frame
            )
        else:
            frame = pd.DataFrame(
                columns=[
                    "rule_key",
                    "rule",
                    "period",
                    "net_mid_pct",
                    "net_high_pct",
                ]
            )

        result_rows.append(
            summarize_rule(
                frame,
                rule,
            )
        )

        if index % 100 == 0:
            print(
                f"Evaluated {index:,}/"
                f"{len(rules):,}"
            )

    results = pd.DataFrame(
        result_rows
    )

    trades = (
        pd.concat(
            all_trades,
            ignore_index=True,
        )
        if all_trades
        else pd.DataFrame()
    )

    eligible = results[
        results["total_trades"]
        >= MIN_TOTAL_TRADES
    ].copy()

    eligible = eligible.sort_values(
        [
            "exploratory_score",
            "combined_avg_net_mid_pct",
            "combined_net_mid_sum_pct",
        ],
        ascending=False,
    ).reset_index(
        drop=True
    )

    top = eligible.head(
        25
    ).copy()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        RESULT_FILE,
        index=False,
    )

    trades.to_csv(
        TRADE_FILE,
        index=False,
    )

    top.to_csv(
        TOP_FILE,
        index=False,
    )

    summary = {
        "purpose":
            "Broad exploratory search across AI confidence and technical-filter combinations.",
        "warning":
            "Validation and test periods are both used for discovery in this experiment. Results are not unbiased proof of profitability.",
        "candidate_count":
            int(
                len(
                    rules
                )
            ),
        "minimum_total_trades_for_ranking":
            MIN_TOTAL_TRADES,
        "confidence_thresholds":
            CONFIDENCE_THRESHOLDS,
        "rsi_rules": [
            None,
            "25/75",
            "30/70",
            "35/65",
            "40/60",
        ],
        "relative_volume_thresholds":
            RELATIVE_VOLUME_THRESHOLDS,
        "filters": [
            "EMA alignment",
            "MACD alignment",
            "Bollinger mean reversion",
            "relative volume",
        ],
        "next_step":
            "Freeze one or more NEW candidates only after reviewing robustness, then evaluate only on genuinely new forward data.",
    }

    SUMMARY_FILE.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    if top.empty:
        print(
            "No candidate met the minimum trade-count requirement."
        )
    else:
        display_columns = [
            "rule",
            "validation_trades",
            "test_trades",
            "combined_win_rate_mid_pct",
            "combined_avg_net_mid_pct",
            "combined_net_mid_sum_pct",
            "combined_net_high_sum_pct",
            "combined_max_drawdown_mid_pct",
            "exploratory_score",
        ]

        print()
        print(
            "TOP EXPLORATORY CANDIDATES"
        )
        print()

        print(
            top[
                display_columns
            ].head(
                15
            ).to_string(
                index=False
            )
        )

    print()
    print(
        "These are discovery results, not a final test."
    )

    print(
        "Do not overwrite the existing frozen V1 candidate."
    )

    print()
    print(
        "Saved:"
    )
    print(
        RESULT_FILE
    )
    print(
        TRADE_FILE
    )
    print(
        TOP_FILE
    )
    print(
        SUMMARY_FILE
    )


if __name__ == "__main__":
    main()
