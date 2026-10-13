import {
  useEffect,
  useState,
} from 'react'

import DashboardPageShell
  from '../../components/layout/DashboardPageShell'

import {
  getDashboardData,
} from '../../services/tradingApi'

import {
  getManualPaperAccount,
} from '../../services/paperTradingApi'

import {
  getMicroPaperStatus,
} from '../../services/microPaperApi'

import type {
  ManualPaperAccount,
  MicroPaperStatus,
  TradingDashboardData,
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


const PerformancePage = () => {
  const [manual, setManual] =
    useState<ManualPaperAccount | null>(null)

  const [micro, setMicro] =
    useState<MicroPaperStatus | null>(null)

  const [strategy, setStrategy] =
    useState<TradingDashboardData | null>(null)

  const [error, setError] =
    useState<string | null>(null)


  useEffect(() => {
    let active = true

    const load = async () => {
      const results =
        await Promise.allSettled([
          getManualPaperAccount(),
          getMicroPaperStatus(),
          getDashboardData('10m'),
        ])

      if (!active) {
        return
      }

      const [
        manualResult,
        microResult,
        strategyResult,
      ] = results

      if (manualResult.status === 'fulfilled') {
        setManual(manualResult.value)
      }

      if (microResult.status === 'fulfilled') {
        setMicro(microResult.value)
      }

      if (strategyResult.status === 'fulfilled') {
        setStrategy(strategyResult.value)
      }

      const failed =
        results.filter(
          (item) => item.status === 'rejected',
        ).length

      setError(
        failed > 0
          ? String(failed)
            + ' performance source(s) are currently unavailable.'
          : null,
      )
    }

    load()

    const timer =
      window.setInterval(load, 10_000)

    return () => {
      active = false
      window.clearInterval(timer)
    }
  }, [])


  return (
    <DashboardPageShell
      eyebrow="Performance"
      title="Paper Trading Performance"
      description="Track manual paper equity, autonomous micro-paper results and the frozen 10-minute strategy's forward evidence. Values are simulated and should not be interpreted as guaranteed future returns."
    >
      {error && (
        <div className="mb-6 rounded-xl border border-amber-400/20 bg-amber-400/[0.05] px-4 py-3 text-sm text-amber-200">
          {error}
        </div>
      )}

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <MetricCard
          label="Manual equity"
          value={
            manual
              ? '$' + money(manual.equityUsdt)
              : '—'
          }
          note="Simulated USDT + BTC market value"
        />

        <MetricCard
          label="Manual P&L"
          value={
            manual
              ? '$' + money(
                  manual.realizedPnlUsdt
                  + manual.unrealizedPnlUsdt,
                )
              : '—'
          }
          note="Realized + unrealized"
        />

        <MetricCard
          label="Micro win rate"
          value={
            micro
              ? micro.winRatePct.toFixed(2) + '%'
              : '—'
          }
          note={
            micro
              ? String(micro.wins)
                + ' wins / '
                + String(micro.losses)
                + ' losses'
              : 'Experimental paper learner'
          }
        />

        <MetricCard
          label="10m mid-cost net"
          value={
            strategy
              ? strategy.netReturn.toFixed(2) + '%'
              : '—'
          }
          note="Forward research evidence"
        />
      </div>

      <div className="mt-6 grid gap-6 xl:grid-cols-2">
        <section className="rounded-2xl border border-white/10 bg-white/[0.025] p-6">
          <p className="text-xs uppercase tracking-[0.2em] text-white/30">
            Manual account
          </p>

          {manual ? (
            <div className="mt-5 grid gap-3 sm:grid-cols-2">
              <SmallMetric
                label="Cash"
                value={'$' + money(manual.cashUsdt)}
              />

              <SmallMetric
                label="BTC holdings"
                value={
                  manual.btcQuantity.toFixed(6)
                  + ' BTC'
                }
              />

              <SmallMetric
                label="BTC market value"
                value={
                  '$'
                  + money(manual.btcMarketValueUsdt)
                }
              />

              <SmallMetric
                label="Unrealized P&L"
                value={
                  '$'
                  + money(manual.unrealizedPnlUsdt)
                }
              />
            </div>
          ) : (
            <p className="mt-4 text-sm text-white/35">
              Manual paper account unavailable.
            </p>
          )}
        </section>

        <section className="rounded-2xl border border-white/10 bg-white/[0.025] p-6">
          <p className="text-xs uppercase tracking-[0.2em] text-white/30">
            10m forward research
          </p>

          {strategy ? (
            <div className="mt-5 space-y-4 text-sm">
              <Row
                label="Decisions recorded"
                value={String(strategy.forwardSignalCount)}
              />

              <Row
                label="Closed forward trades"
                value={String(strategy.closedForwardTrades)}
              />

              <Row
                label="Primary modeled cost"
                value={
                  strategy.primaryCostPct.toFixed(2)
                  + '%'
                }
              />

              <Row
                label="Research status"
                value={strategy.researchStatus}
              />
            </div>
          ) : (
            <p className="mt-4 text-sm text-white/35">
              10m strategy evidence unavailable.
            </p>
          )}
        </section>
      </div>

      <section className="mt-6 overflow-hidden rounded-2xl border border-white/10 bg-white/[0.025]">
        <div className="border-b border-white/5 px-6 py-5">
          <p className="text-xs uppercase tracking-[0.2em] text-white/30">
            Recent autonomous micro trades
          </p>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full min-w-[760px] text-left text-sm">
            <thead className="text-xs uppercase tracking-wider text-white/25">
              <tr>
                <th className="px-6 py-4">Time</th>
                <th className="px-6 py-4">Event</th>
                <th className="px-6 py-4">Side</th>
                <th className="px-6 py-4">Notional</th>
                <th className="px-6 py-4">Net return</th>
                <th className="px-6 py-4">P&amp;L</th>
              </tr>
            </thead>

            <tbody>
              {micro?.recentTrades
                .slice(0, 10)
                .map((trade) => (
                  <tr
                    key={
                      trade.timestamp
                      + trade.event
                      + trade.side
                    }
                    className="border-t border-white/5"
                  >
                    <td className="px-6 py-4 text-white/40">
                      {new Date(
                        trade.timestamp,
                      ).toLocaleString()}
                    </td>

                    <td className="px-6 py-4">
                      {trade.event}
                    </td>

                    <td className="px-6 py-4">
                      {trade.side}
                    </td>

                    <td className="px-6 py-4">
                      {'$' + money(trade.notionalUsdt)}
                    </td>

                    <td className="px-6 py-4">
                      {trade.netReturnPct.toFixed(3)}%
                    </td>

                    <td className="px-6 py-4">
                      {'$' + money(trade.pnlUsdt)}
                    </td>
                  </tr>
                ))}

              {(
                !micro
                || micro.recentTrades.length === 0
              ) && (
                <tr>
                  <td
                    colSpan={6}
                    className="px-6 py-8 text-center text-white/30"
                  >
                    No autonomous micro trades are available.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </section>
    </DashboardPageShell>
  )
}


const MetricCard = ({
  label,
  value,
  note,
}: {
  label: string
  value: string
  note: string
}) => (
  <div className="rounded-2xl border border-white/10 bg-white/[0.025] p-5">
    <p className="text-xs uppercase tracking-wider text-white/25">
      {label}
    </p>

    <p className="mt-2 text-2xl font-semibold">
      {value}
    </p>

    <p className="mt-1 text-xs text-white/30">
      {note}
    </p>
  </div>
)


const SmallMetric = ({
  label,
  value,
}: {
  label: string
  value: string
}) => (
  <div className="rounded-xl border border-white/5 bg-black/10 p-4">
    <p className="text-xs text-white/30">
      {label}
    </p>

    <p className="mt-2 font-medium">
      {value}
    </p>
  </div>
)


const Row = ({
  label,
  value,
}: {
  label: string
  value: string
}) => (
  <div className="flex justify-between gap-6 border-b border-white/5 pb-3 last:border-b-0">
    <span className="text-white/35">
      {label}
    </span>

    <span className="max-w-[60%] text-right text-white/75">
      {value}
    </span>
  </div>
)


export default PerformancePage
