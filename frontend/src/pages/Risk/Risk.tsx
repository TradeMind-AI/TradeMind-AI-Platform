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


const RiskPage = () => {
  const [manual, setManual] =
    useState<ManualPaperAccount | null>(null)

  const [micro, setMicro] =
    useState<MicroPaperStatus | null>(null)

  const [strategy, setStrategy] =
    useState<TradingDashboardData | null>(null)

  const [lastUpdated, setLastUpdated] =
    useState<string>('')


  const load = async () => {
    const results =
      await Promise.allSettled([
        getManualPaperAccount(),
        getMicroPaperStatus(),
        getDashboardData('10m'),
      ])

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

    setLastUpdated(
      new Date().toLocaleTimeString(),
    )
  }


  useEffect(() => {
    load()

    const timer =
      window.setInterval(load, 10_000)

    return () =>
      window.clearInterval(timer)
  }, [])


  const manualExposurePct =
    manual && manual.equityUsdt > 0
      ? (
        manual.btcMarketValueUsdt
        / manual.equityUsdt
        * 100
      )
      : 0

  const microDrawdownPct =
    micro
      ? Math.max(
          0,
          (
            10000
            - micro.equityUsdt
          )
          / 10000
          * 100,
        )
      : 0

  const overallRisk =
    microDrawdownPct >= 5
    || manualExposurePct >= 75
      ? 'HIGH'
      : (
        microDrawdownPct >= 2
        || manualExposurePct >= 40
      )
        ? 'MODERATE'
        : 'LOW'


  return (
    <DashboardPageShell
      eyebrow="Risk"
      title="Paper Risk Monitor"
      description="Monitor simulated BTC exposure, paper-account drawdown, transaction-cost assumptions and experimental strategy warnings. These checks are informational safeguards for the paper environment."
    >
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div
          className={[
            'rounded-full border px-4 py-2 text-sm font-semibold',
            overallRisk === 'HIGH'
              ? 'border-red-400/30 bg-red-400/[0.08] text-red-300'
              : overallRisk === 'MODERATE'
                ? 'border-amber-400/30 bg-amber-400/[0.08] text-amber-200'
                : 'border-emerald-400/30 bg-emerald-400/[0.08] text-emerald-300',
          ].join(' ')}
        >
          Current paper risk: {overallRisk}
        </div>

        <button
          type="button"
          onClick={load}
          className="rounded-lg border border-white/10 bg-white/[0.03] px-4 py-2 text-sm text-white/60 transition hover:text-white"
        >
          Refresh risk checks
        </button>
      </div>

      <p className="mt-2 text-xs text-white/25">
        Last updated: {lastUpdated || 'waiting'}
      </p>

      <div className="mt-6 grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <RiskMetric
          label="Manual BTC exposure"
          value={
            manualExposurePct.toFixed(2)
            + '%'
          }
          note="BTC market value as a share of manual paper equity"
        />

        <RiskMetric
          label="Micro drawdown"
          value={
            microDrawdownPct.toFixed(2)
            + '%'
          }
          note="Measured from the $10,000 experimental starting equity"
        />

        <RiskMetric
          label="10m primary cost"
          value={
            strategy
              ? strategy.primaryCostPct.toFixed(2)
                + '%'
              : '—'
          }
          note="Modeled all-in round-trip research cost"
        />

        <RiskMetric
          label="10m stress cost"
          value={
            strategy
              ? strategy.stressCostPct.toFixed(2)
                + '%'
              : '—'
          }
          note="Higher-cost robustness assumption"
        />
      </div>

      <div className="mt-6 grid gap-6 xl:grid-cols-2">
        <section className="rounded-2xl border border-white/10 bg-white/[0.025] p-6">
          <p className="text-xs uppercase tracking-[0.2em] text-white/30">
            Live risk checks
          </p>

          <div className="mt-5 space-y-3">
            <Check
              ok={manualExposurePct < 75}
              label="Manual account concentration"
              detail={
                manualExposurePct < 75
                  ? 'BTC exposure is below the 75% warning level.'
                  : 'BTC exposure is above the 75% warning level.'
              }
            />

            <Check
              ok={microDrawdownPct < 5}
              label="Micro-paper drawdown"
              detail={
                microDrawdownPct < 5
                  ? 'Experimental drawdown is below 5%.'
                  : 'Experimental drawdown has reached at least 5%; the learner should not be treated as deployable.'
              }
            />

            <Check
              ok={
                !micro?.running
              }
              label="Experimental micro learner"
              detail={
                micro?.running
                  ? 'Micro learner data is active. Treat it as research only.'
                  : 'Micro learner is not reporting as live.'
              }
            />

            <Check
              ok={
                Boolean(
                  strategy
                  && strategy.researchStatus,
                )
              }
              label="Frozen 10m research status"
              detail={
                strategy
                  ? strategy.researchStatus
                  : '10m strategy metadata unavailable.'
              }
            />
          </div>
        </section>

        <section className="rounded-2xl border border-white/10 bg-white/[0.025] p-6">
          <p className="text-xs uppercase tracking-[0.2em] text-white/30">
            Risk principles
          </p>

          <div className="mt-5 space-y-4 text-sm leading-6 text-white/45">
            <p>
              <span className="font-medium text-white/75">
                Paper only.
              </span>{' '}
              No control on this page sends a real Binance order.
            </p>

            <p>
              <span className="font-medium text-white/75">
                Costs matter.
              </span>{' '}
              A model can predict direction correctly and still lose after spread, fees and slippage.
            </p>

            <p>
              <span className="font-medium text-white/75">
                Confidence is not accuracy.
              </span>{' '}
              High model probability must be checked against unseen outcomes.
            </p>

            <p>
              <span className="font-medium text-white/75">
                Frozen evidence.
              </span>{' '}
              Promising candidates should be forward-tested instead of repeatedly retuned on previously viewed validation periods.
            </p>
          </div>
        </section>
      </div>
    </DashboardPageShell>
  )
}


const RiskMetric = ({
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

    <p className="mt-1 text-xs leading-5 text-white/30">
      {note}
    </p>
  </div>
)


const Check = ({
  ok,
  label,
  detail,
}: {
  ok: boolean
  label: string
  detail: string
}) => (
  <div className="rounded-xl border border-white/5 bg-black/10 p-4">
    <div className="flex items-start gap-3">
      <span
        className={[
          'mt-1 h-2.5 w-2.5 shrink-0 rounded-full',
          ok
            ? 'bg-emerald-400'
            : 'bg-amber-400',
        ].join(' ')}
      />

      <div>
        <p className="font-medium text-white/75">
          {label}
        </p>

        <p className="mt-1 text-xs leading-5 text-white/35">
          {detail}
        </p>
      </div>
    </div>
  </div>
)


export default RiskPage
