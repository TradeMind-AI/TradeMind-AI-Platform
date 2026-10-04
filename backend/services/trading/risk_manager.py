from dataclasses import dataclass

import pandas as pd


STARTING_BALANCE = 10_000.0

RISK_PER_TRADE_PCT = 1.0
MAX_POSITION_PCT = 25.0

STOP_LOSS_PCT = 0.50
TAKE_PROFIT_PCT = 1.00

MAX_DAILY_LOSS_PCT = 3.0


@dataclass
class RiskDecision:
    allowed: bool
    reason: str

    balance: float = 0.0
    risk_amount: float = 0.0
    position_notional: float = 0.0
    quantity: float = 0.0

    stop_loss_price: float = 0.0
    take_profit_price: float = 0.0


def get_current_balance(
    paper_log: pd.DataFrame | None,
) -> float:
    """
    Return paper account balance from realized P&L.
    """

    if paper_log is None or paper_log.empty:
        return STARTING_BALANCE

    if "account_pnl" not in paper_log.columns:
        return STARTING_BALANCE

    pnl = pd.to_numeric(
        paper_log["account_pnl"],
        errors="coerce",
    ).fillna(0.0)

    return STARTING_BALANCE + pnl.sum()


def get_daily_realized_pnl(
    paper_log: pd.DataFrame | None,
    timestamp,
) -> float:
    """
    Calculate realized account P&L for the UTC day
    containing the proposed trade.
    """

    if paper_log is None or paper_log.empty:
        return 0.0

    if (
        "timestamp" not in paper_log.columns
        or "account_pnl" not in paper_log.columns
    ):
        return 0.0

    log = paper_log.copy()

    log["timestamp"] = pd.to_datetime(
        log["timestamp"],
        utc=True,
        errors="coerce",
    )

    log["account_pnl"] = pd.to_numeric(
        log["account_pnl"],
        errors="coerce",
    ).fillna(0.0)

    timestamp = pd.to_datetime(
        timestamp,
        utc=True,
    )

    same_day = (
        log["timestamp"].dt.date
        == timestamp.date()
    )

    return float(
        log.loc[
            same_day,
            "account_pnl",
        ].sum()
    )


def evaluate_trade(
    signal: str,
    entry_price: float,
    timestamp,
    paper_log: pd.DataFrame | None,
) -> RiskDecision:

    balance = get_current_balance(
        paper_log
    )

    if signal not in {"BUY", "SELL"}:
        return RiskDecision(
            allowed=False,
            reason="No directional trade signal.",
            balance=balance,
        )

    if balance <= 0:
        return RiskDecision(
            allowed=False,
            reason="Paper account balance is depleted.",
            balance=balance,
        )

    daily_pnl = get_daily_realized_pnl(
        paper_log,
        timestamp,
    )

    max_daily_loss = (
        balance
        * MAX_DAILY_LOSS_PCT
        / 100
    )

    if daily_pnl <= -max_daily_loss:
        return RiskDecision(
            allowed=False,
            reason=(
                "Maximum daily loss limit reached."
            ),
            balance=balance,
        )

    risk_amount = (
        balance
        * RISK_PER_TRADE_PCT
        / 100
    )

    stop_decimal = (
        STOP_LOSS_PCT / 100
    )

    # Position size implied by risk limit.
    risk_based_notional = (
        risk_amount
        / stop_decimal
    )

    # Hard cap prevents excessive leverage/exposure.
    maximum_notional = (
        balance
        * MAX_POSITION_PCT
        / 100
    )

    position_notional = min(
        risk_based_notional,
        maximum_notional,
    )

    quantity = (
        position_notional
        / entry_price
    )

    if signal == "BUY":

        stop_loss_price = (
            entry_price
            * (
                1
                - STOP_LOSS_PCT / 100
            )
        )

        take_profit_price = (
            entry_price
            * (
                1
                + TAKE_PROFIT_PCT / 100
            )
        )

    else:

        stop_loss_price = (
            entry_price
            * (
                1
                + STOP_LOSS_PCT / 100
            )
        )

        take_profit_price = (
            entry_price
            * (
                1
                - TAKE_PROFIT_PCT / 100
            )
        )

    return RiskDecision(
        allowed=True,
        reason="Trade approved.",
        balance=balance,
        risk_amount=risk_amount,
        position_notional=position_notional,
        quantity=quantity,
        stop_loss_price=stop_loss_price,
        take_profit_price=take_profit_price,
    )