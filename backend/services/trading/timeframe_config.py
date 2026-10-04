from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class TimeframeConfig:
    key: str
    pandas_rule: str
    minutes: int
    model_file: Path
    feature_file: Path
    paper_log: Path
    market_file: Path


def _model_file(key: str) -> Path:
    if key == "1h":
        return (
            PROJECT_ROOT
            / "backend"
            / "models"
            / "locked_xgboost_bundle.joblib"
        )

    return (
        PROJECT_ROOT
        / "backend"
        / "models"
        / f"locked_xgboost_{key}_bundle.joblib"
    )


def _feature_file(key: str) -> Path:
    if key == "1h":
        return (
            PROJECT_ROOT
            / "data"
            / "processed"
            / "btcusdt_1h_live_features.csv"
        )

    return (
        PROJECT_ROOT
        / "data"
        / "processed"
        / f"btcusdt_{key}_live_features.csv"
    )


def _paper_log(key: str) -> Path:
    if key == "1h":
        return (
            PROJECT_ROOT
            / "data"
            / "processed"
            / "paper_trades.csv"
        )

    if key == "10m":
        return (
            PROJECT_ROOT
            / "data"
            / "processed"
            / "paper_forward_10m.csv"
        )

    return (
        PROJECT_ROOT
        / "data"
        / "processed"
        / f"paper_trades_{key}.csv"
    )


def _market_file(key: str) -> Path:
    if key == "1h":
        return (
            PROJECT_ROOT
            / "data"
            / "processed"
            / "btcusdt_1h_clean.csv"
        )

    return (
        PROJECT_ROOT
        / "data"
        / "processed"
        / f"btcusdt_{key}.csv"
    )


TIMEFRAMES = {
    "1m": TimeframeConfig(
        key="1m",
        pandas_rule="1min",
        minutes=1,
        model_file=_model_file("1m"),
        feature_file=_feature_file("1m"),
        paper_log=_paper_log("1m"),
        market_file=_market_file("1m"),
    ),
    "2m": TimeframeConfig(
        key="2m",
        pandas_rule="2min",
        minutes=2,
        model_file=_model_file("2m"),
        feature_file=_feature_file("2m"),
        paper_log=_paper_log("2m"),
        market_file=_market_file("2m"),
    ),
    "5m": TimeframeConfig(
        key="5m",
        pandas_rule="5min",
        minutes=5,
        model_file=_model_file("5m"),
        feature_file=_feature_file("5m"),
        paper_log=_paper_log("5m"),
        market_file=_market_file("5m"),
    ),
    "10m": TimeframeConfig(
        key="10m",
        pandas_rule="10min",
        minutes=10,
        model_file=_model_file("10m"),
        feature_file=_feature_file("10m"),
        paper_log=_paper_log("10m"),
        market_file=_market_file("10m"),
    ),
    "20m": TimeframeConfig(
        key="20m",
        pandas_rule="20min",
        minutes=20,
        model_file=_model_file("20m"),
        feature_file=_feature_file("20m"),
        paper_log=_paper_log("20m"),
        market_file=_market_file("20m"),
    ),
    "30m": TimeframeConfig(
        key="30m",
        pandas_rule="30min",
        minutes=30,
        model_file=_model_file("30m"),
        feature_file=_feature_file("30m"),
        paper_log=_paper_log("30m"),
        market_file=_market_file("30m"),
    ),
    "1h": TimeframeConfig(
        key="1h",
        pandas_rule="1h",
        minutes=60,
        model_file=_model_file("1h"),
        feature_file=_feature_file("1h"),
        paper_log=_paper_log("1h"),
        market_file=_market_file("1h"),
    ),
}


SUPPORTED_TIMEFRAMES = tuple(TIMEFRAMES.keys())


def get_timeframe_config(key: str) -> TimeframeConfig:
    normalized = key.strip().lower()

    if normalized not in TIMEFRAMES:
        supported = ", ".join(SUPPORTED_TIMEFRAMES)
        raise ValueError(
            f"Unsupported timeframe '{key}'. "
            f"Supported values: {supported}"
        )

    return TIMEFRAMES[normalized]
