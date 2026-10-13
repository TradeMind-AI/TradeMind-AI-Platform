import {
  useEffect,
  useState,
} from 'react'

import DashboardPageShell
  from '../../components/layout/DashboardPageShell'

import RecentSignals
  from '../../components/dashboard/RecentSignals'

import SignalCard
  from '../../components/dashboard/SignalCard'

import {
  getDashboardData,
} from '../../services/tradingApi'

import type {
  StrategyTimeframe,
  TradingDashboardData,
} from '../../types/trading'


const TIMEFRAMES: StrategyTimeframe[] = [
  '1m',
  '2m',
  '5m',
  '10m',
  '20m',
  '30m',
  '1h',
]


const SignalsPage = () => {
  const [
    timeframe,
    setTimeframe,
  ] = useState<StrategyTimeframe>(
    '10m',
  )

  const [
    data,
    setData,
  ] = useState<TradingDashboardData | null>(
    null,
  )

  const [
    error,
    setError,
  ] = useState<string | null>(
    null,
  )

  const [
    loading,
    setLoading,
  ] = useState(true)


  useEffect(() => {
    let active = true

    const load = async () => {
      try {
        setLoading(true)

        const next =
          await getDashboardData(
            timeframe,
          )

        if (active) {
          setData(next)
          setError(null)
        }
      } catch (err) {
        if (active) {
          setData(null)
          setError(
            err instanceof Error
              ? err.message
              : 'Failed to load signals',
          )
        }
      } finally {
        if (active) {
          setLoading(false)
        }
      }
    }

    load()

    const timer =
      window.setInterval(
        load,
        15_000,
      )

    return () => {
      active = false
      window.clearInterval(timer)
    }
  }, [timeframe])


  return (
    <DashboardPageShell
      eyebrow="Signals"
      title="AI Trading Signals"
      description="Inspect the selected strategy's latest prediction, class probabilities, confidence and recent paper decisions. This page reports model output; it does not place real exchange orders."
    >
      <div className="flex flex-wrap gap-2">
        {TIMEFRAMES.map(
          (item) => (
            <button
              key={item}
              type="button"
              onClick={() =>
                setTimeframe(item)
              }
              className={[
                'rounded-lg border px-4 py-2 text-sm transition',
                item === timeframe
                  ? 'border-white/30 bg-white text-black'
                  : 'border-white/10 bg-white/[0.03] text-white/55 hover:text-white',
              ].join(' ')}
            >
              {item}
            </button>
          ),
        )}
      </div>

      {loading && (
        <div className="mt-6 rounded-2xl border border-white/10 bg-white/[0.025] p-6 text-sm text-white/40">
          Loading {timeframe} model signals...
        </div>
      )}

      {!loading && error && (
        <div className="mt-6 rounded-2xl border border-amber-400/20 bg-amber-400/[0.05] p-6">
          <p className="font-medium text-amber-200">
            {timeframe} signal model is not ready
          </p>

          <p className="mt-2 text-sm text-white/45">
            {error}
          </p>
        </div>
      )}

      {!loading && data && (
        <>
          <div className="mt-6 grid gap-6 xl:grid-cols-[1.2fr_0.8fr]">
            <SignalCard
              prediction={
                data.prediction
              }
              confidence={
                data.confidence
              }
              signal={
                data.signal
              }
              probabilities={
                data.probabilities
              }
            />

            <section className="rounded-2xl border border-white/10 bg-white/[0.025] p-6">
              <p className="text-xs uppercase tracking-[0.2em] text-white/30">
                Signal context
              </p>

              <div className="mt-5 space-y-4 text-sm">
                <div className="flex justify-between border-b border-white/5 pb-3">
                  <span className="text-white/35">
                    Model
                  </span>
                  <span className="text-white/75">
                    {data.modelName}
                  </span>
                </div>

                <div className="flex justify-between border-b border-white/5 pb-3">
                  <span className="text-white/35">
                    Forecast horizon
                  </span>
                  <span className="text-white/75">
                    {data.forecastHorizonMinutes} min
                  </span>
                </div>

                <div className="flex justify-between border-b border-white/5 pb-3">
                  <span className="text-white/35">
                    Target move
                  </span>
                  <span className="text-white/75">
                    ±{data.targetThresholdPct.toFixed(2)}%
                  </span>
                </div>

                <div className="flex justify-between">
                  <span className="text-white/35">
                    Confidence gate
                  </span>
                  <span className="text-white/75">
                    {(data.confidenceThreshold * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            </section>
          </div>

          <div className="mt-6">
            <RecentSignals
              signals={
                data.recentSignals
              }
            />
          </div>
        </>
      )}
    </DashboardPageShell>
  )
}


export default SignalsPage
