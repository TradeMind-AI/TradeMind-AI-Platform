import {
  useEffect,
  useState,
} from 'react'

import {
  getMicroPaperStatus,
} from '../../services/microPaperApi'

import type {
  MicroPaperStatus,
} from '../../types/trading'


const money = (
  value: number,
) => value.toLocaleString(
  'en-US',
  {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  },
)


const pct = (
  value: number,
  digits = 2,
) => value.toFixed(
  digits,
) + '%'


const signalLabel = (
  value: number,
) => {
  if (value > 0) {
    return 'UP'
  }

  if (value < 0) {
    return 'DOWN'
  }

  return 'NEUTRAL'
}


const MicroPaperMonitor = () => {
  const [
    status,
    setStatus,
  ] = useState<MicroPaperStatus | null>(
    null,
  )

  const [
    error,
    setError,
  ] = useState<string | null>(
    null,
  )


  useEffect(() => {
    let active = true

    const refresh = async () => {
      try {
        const next =
          await getMicroPaperStatus()

        if (active) {
          setStatus(next)
          setError(null)
        }
      } catch (err) {
        if (active) {
          setError(
            err instanceof Error
              ? err.message
              : 'Failed to load micro paper learner',
          )
        }
      }
    }

    refresh()

    const timer =
      window.setInterval(
        refresh,
        5_000,
      )

    return () => {
      active = false
      window.clearInterval(
        timer,
      )
    }
  }, [])


  const latest =
    status?.latest

  const position =
    status?.position


  return (
    <section className="rounded-2xl border border-white/10 bg-white/[0.025] p-6">
      <div className="flex flex-col gap-4 border-b border-white/5 pb-5 md:flex-row md:items-start md:justify-between">
        <div>
          <p className="text-xs uppercase tracking-[0.2em] text-white/30">
            Micro AI Paper Trader
          </p>

          <h2 className="mt-2 text-xl font-semibold">
            Online learner + autonomous paper trades
          </h2>

          <p className="mt-2 max-w-3xl text-sm leading-6 text-white/35">
            This is the experimental 5-second microstructure learner. It trades only with simulated funds and learns from future market outcomes.
          </p>
        </div>

        <span
          className={[
            'rounded-md border px-2.5 py-1 text-[10px] uppercase tracking-wider',
            status?.running
              ? 'border-emerald-400/20 bg-emerald-400/[0.06] text-emerald-300'
              : 'border-white/10 bg-white/[0.03] text-white/35',
          ].join(' ')}
        >
          {status?.running
            ? 'data live'
            : 'waiting'}
        </span>
      </div>

      {error && (
        <div className="mt-4 rounded-lg border border-amber-400/20 bg-amber-400/[0.05] px-3 py-2 text-xs text-amber-200">
          {error}
        </div>
      )}

      {status && (
        <>
          <div className="mt-5 grid gap-3 sm:grid-cols-2 xl:grid-cols-5">
            <div className="rounded-xl border border-white/8 bg-black/10 p-4">
              <p className="text-[10px] uppercase tracking-wider text-white/25">
                Micro equity
              </p>
              <p className="mt-2 text-lg font-semibold">
                {'$' + money(status.equityUsdt)}
              </p>
            </div>

            <div className="rounded-xl border border-white/8 bg-black/10 p-4">
              <p className="text-[10px] uppercase tracking-wider text-white/25">
                Closed trades
              </p>
              <p className="mt-2 text-lg font-semibold">
                {status.closedTrades}
              </p>
            </div>

            <div className="rounded-xl border border-white/8 bg-black/10 p-4">
              <p className="text-[10px] uppercase tracking-wider text-white/25">
                Win rate
              </p>
              <p className="mt-2 text-lg font-semibold">
                {pct(status.winRatePct)}
              </p>
              <p className="mt-1 text-xs text-white/25">
                {status.wins} wins · {status.losses} losses
              </p>
            </div>

            <div className="rounded-xl border border-white/8 bg-black/10 p-4">
              <p className="text-[10px] uppercase tracking-wider text-white/25">
                Realized P&amp;L
              </p>
              <p className="mt-2 text-lg font-semibold">
                {'$' + money(status.realizedPnlUsdt)}
              </p>
            </div>

            <div className="rounded-xl border border-white/8 bg-black/10 p-4">
              <p className="text-[10px] uppercase tracking-wider text-white/25">
                Learning samples
              </p>
              <p className="mt-2 text-lg font-semibold">
                {latest?.learningSamples ?? 0}
              </p>
            </div>
          </div>

          <div className="mt-5 grid gap-5 xl:grid-cols-[0.9fr_1.1fr]">
            <div className="rounded-xl border border-white/8 bg-black/10 p-5">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <p className="text-[10px] uppercase tracking-[0.18em] text-white/25">
                    Current model decision
                  </p>

                  <p className="mt-2 text-2xl font-semibold">
                    {latest
                      ? signalLabel(
                          latest.prediction,
                        )
                      : 'WAITING'}
                  </p>
                </div>

                <div className="text-right">
                  <p className="text-xs text-white/30">
                    Confidence
                  </p>

                  <p className="mt-1 text-xl font-semibold">
                    {latest
                      ? pct(
                          latest.confidence
                          * 100,
                          1,
                        )
                      : '0.0%'}
                  </p>
                </div>
              </div>

              {latest && (
                <>
                  <div className="mt-5 grid grid-cols-3 gap-2 text-center text-xs">
                    <div className="rounded-lg border border-red-400/10 bg-red-400/[0.03] p-3">
                      <p className="text-white/25">
                        DOWN
                      </p>
                      <p className="mt-1 font-medium text-red-300">
                        {pct(
                          latest.probDown
                          * 100,
                          1,
                        )}
                      </p>
                    </div>

                    <div className="rounded-lg border border-white/10 bg-white/[0.02] p-3">
                      <p className="text-white/25">
                        NEUTRAL
                      </p>
                      <p className="mt-1 font-medium text-white/60">
                        {pct(
                          latest.probNeutral
                          * 100,
                          1,
                        )}
                      </p>
                    </div>

                    <div className="rounded-lg border border-emerald-400/10 bg-emerald-400/[0.03] p-3">
                      <p className="text-white/25">
                        UP
                      </p>
                      <p className="mt-1 font-medium text-emerald-300">
                        {pct(
                          latest.probUp
                          * 100,
                          1,
                        )}
                      </p>
                    </div>
                  </div>

                  <div className="mt-5 rounded-lg border border-white/5 p-4 text-xs">
                    <div className="flex items-center justify-between gap-4">
                      <span className="text-white/30">
                        BTC price
                      </span>
                      <span className="text-white/70">
                        {'$' + money(latest.price)}
                      </span>
                    </div>

                    <div className="mt-2 flex items-center justify-between gap-4">
                      <span className="text-white/30">
                        Decision source
                      </span>
                      <span className="text-white/70">
                        {latest.decisionSource || 'WAITING'}
                      </span>
                    </div>

                    <div className="mt-2 flex items-center justify-between gap-4">
                      <span className="text-white/30">
                        Trade direction
                      </span>
                      <span className="text-white/70">
                        {signalLabel(
                          latest.tradeDirection,
                        )}
                      </span>
                    </div>
                  </div>
                </>
              )}

              <div className="mt-5 border-t border-white/5 pt-4">
                <p className="text-[10px] uppercase tracking-[0.18em] text-white/25">
                  Open micro position
                </p>

                {!position && (
                  <p className="mt-2 text-sm text-white/35">
                    Flat — no autonomous paper position is open.
                  </p>
                )}

                {position && (
                  <div className="mt-3 grid grid-cols-2 gap-3 text-xs">
                    <div>
                      <p className="text-white/25">
                        Side
                      </p>
                      <p className={
                        position.direction > 0
                          ? 'mt-1 font-medium text-emerald-300'
                          : 'mt-1 font-medium text-red-300'
                      }>
                        {position.direction > 0
                          ? 'LONG'
                          : 'SHORT'}
                      </p>
                    </div>

                    <div>
                      <p className="text-white/25">
                        Entry
                      </p>
                      <p className="mt-1 text-white/70">
                        {'$' + money(position.entry_price)}
                      </p>
                    </div>

                    <div>
                      <p className="text-white/25">
                        Notional
                      </p>
                      <p className="mt-1 text-white/70">
                        {'$' + money(position.notional_usdt)}
                      </p>
                    </div>

                    <div>
                      <p className="text-white/25">
                        Entry confidence
                      </p>
                      <p className="mt-1 text-white/70">
                        {pct(
                          position.model_confidence
                          * 100,
                          1,
                        )}
                      </p>
                    </div>
                  </div>
                )}
              </div>
            </div>

            <div className="rounded-xl border border-white/8 bg-black/10 p-5">
              <p className="text-[10px] uppercase tracking-[0.18em] text-white/25">
                Live microstructure
              </p>

              {latest ? (
                <div className="mt-4 grid gap-3 sm:grid-cols-2">
                  <div className="rounded-lg border border-white/5 p-3">
                    <p className="text-xs text-white/30">
                      Spot depth imbalance
                    </p>
                    <p className="mt-1 font-medium">
                      {latest.spotDepthImbalance.toFixed(3)}
                    </p>
                  </div>

                  <div className="rounded-lg border border-white/5 p-3">
                    <p className="text-xs text-white/30">
                      Spot trade flow
                    </p>
                    <p className="mt-1 font-medium">
                      {latest.spotTradeFlowImbalance.toFixed(3)}
                    </p>
                  </div>

                  <div className="rounded-lg border border-white/5 p-3">
                    <p className="text-xs text-white/30">
                      Spot velocity
                    </p>
                    <p className="mt-1 font-medium">
                      {latest.spotVelocityBps.toFixed(2)} bps
                    </p>
                  </div>

                  <div className="rounded-lg border border-white/5 p-3">
                    <p className="text-xs text-white/30">
                      Futures depth imbalance
                    </p>
                    <p className="mt-1 font-medium">
                      {latest.futuresDepthImbalance.toFixed(3)}
                    </p>
                  </div>

                  <div className="rounded-lg border border-white/5 p-3">
                    <p className="text-xs text-white/30">
                      Futures trade flow
                    </p>
                    <p className="mt-1 font-medium">
                      {latest.futuresTradeFlowImbalance.toFixed(3)}
                    </p>
                  </div>

                  <div className="rounded-lg border border-white/5 p-3">
                    <p className="text-xs text-white/30">
                      Futures velocity
                    </p>
                    <p className="mt-1 font-medium">
                      {latest.futuresVelocityBps.toFixed(2)} bps
                    </p>
                  </div>

                  <div className="rounded-lg border border-white/5 p-3">
                    <p className="text-xs text-white/30">
                      Funding
                    </p>
                    <p className="mt-1 font-medium">
                      {(latest.fundingRate * 100).toFixed(4)}%
                    </p>
                  </div>

                  <div className="rounded-lg border border-white/5 p-3">
                    <p className="text-xs text-white/30">
                      Open interest change
                    </p>
                    <p className="mt-1 font-medium">
                      {pct(
                        latest.openInterestChangePct,
                        4,
                      )}
                    </p>
                  </div>
                </div>
              ) : (
                <p className="mt-4 text-sm text-white/30">
                  Waiting for the learner to write its first feature snapshot.
                </p>
              )}
            </div>
          </div>

          <div className="mt-5 rounded-xl border border-white/8 bg-black/10 p-5">
            <div className="flex items-center justify-between gap-4">
              <p className="text-[10px] uppercase tracking-[0.18em] text-white/25">
                Recent autonomous micro trades
              </p>

              <span className="text-xs text-white/25">
                paper only
              </span>
            </div>

            <div className="mt-4 space-y-2">
              {status.recentTrades.length === 0 && (
                <p className="py-5 text-sm text-white/25">
                  No micro trades recorded yet.
                </p>
              )}

              {status.recentTrades.slice(0, 10).map(
                (trade) => (
                  <div
                    key={
                      trade.timestamp
                      + trade.event
                      + trade.side
                    }
                    className="grid gap-2 rounded-lg border border-white/5 px-3 py-3 text-xs sm:grid-cols-[0.7fr_0.7fr_1fr_1fr_1fr_1.2fr]"
                  >
                    <span className={
                      trade.side === 'LONG'
                        ? 'font-medium text-emerald-300'
                        : 'font-medium text-red-300'
                    }>
                      {trade.event} {trade.side}
                    </span>

                    <span className="text-white/45">
                      {'$' + money(trade.notionalUsdt)}
                    </span>

                    <span className="text-white/35">
                      {'Entry $' + money(trade.entryPrice)}
                    </span>

                    <span className="text-white/35">
                      {trade.exitPrice > 0
                        ? 'Exit $' + money(trade.exitPrice)
                        : 'Open'}
                    </span>

                    <span className={
                      trade.pnlUsdt > 0
                        ? 'text-emerald-300'
                        : trade.pnlUsdt < 0
                          ? 'text-red-300'
                          : 'text-white/35'
                    }>
                      {trade.event === 'CLOSE'
                        ? '$' + money(trade.pnlUsdt)
                        : trade.reason}
                    </span>

                    <span className="text-right text-white/25">
                      {new Date(
                        trade.timestamp,
                      ).toLocaleTimeString()}
                    </span>
                  </div>
                ),
              )}
            </div>
          </div>
        </>
      )}
    </section>
  )
}


export default MicroPaperMonitor
