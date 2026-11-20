import {
  useEffect,
  useState,
} from 'react'

import {
  getTenMinuteV2Comparison,
} from '../../services/tenMinuteResearchApi'

import type {
  TenMinuteV2Comparison,
} from '../../types/trading'


const CandidateMetrics = ({
  title,
  data,
}: {
  title: string
  data: TenMinuteV2Comparison['v1']
}) => {
  return (
    <div className="rounded-xl border border-white/5 bg-black/10 p-4">
      <div className="flex items-center justify-between gap-3">
        <p className="font-medium text-white/80">
          {title}
        </p>

        <span className="text-xs text-white/30">
          {data.available ? 'collecting' : 'not started'}
        </span>
      </div>

      <div className="mt-4 grid grid-cols-2 gap-3 text-sm">
        <div>
          <p className="text-white/30">
            Decisions
          </p>
          <p className="mt-1 text-lg font-semibold">
            {data.decisions}
          </p>
        </div>

        <div>
          <p className="text-white/30">
            Closed trades
          </p>
          <p className="mt-1 text-lg font-semibold">
            {data.closedTrades}
          </p>
        </div>

        <div>
          <p className="text-white/30">
            Win rate
          </p>
          <p className="mt-1 text-lg font-semibold">
            {data.winRatePct.toFixed(2)}%
          </p>
        </div>

        <div>
          <p className="text-white/30">
            Mid-cost net
          </p>
          <p className="mt-1 text-lg font-semibold">
            {data.netReturnPct.toFixed(2)}%
          </p>
        </div>
      </div>

      <div className="mt-4 border-t border-white/5 pt-3 text-xs text-white/35">
        Latest: {data.latestSignal || 'HOLD'} · {data.latestSignalTime || 'no signal yet'}
      </div>
    </div>
  )
}


const TenMinuteV2ComparisonPanel = () => {
  const [
    data,
    setData,
  ] =
    useState<TenMinuteV2Comparison | null>(
      null,
    )

  const [
    error,
    setError,
  ] =
    useState<string | null>(
      null,
    )

  useEffect(() => {
    let active = true

    const load = async () => {
      try {
        const next =
          await getTenMinuteV2Comparison()

        if (active) {
          setData(next)
          setError(null)
        }
      } catch (err) {
        if (active) {
          setError(
            err instanceof Error
              ? err.message
              : 'Failed to load V2 comparison',
          )
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
  }, [])

  return (
    <section className="rounded-2xl border border-white/10 bg-white/[0.025] p-6">
      <div>
        <p className="text-xs uppercase tracking-[0.2em] text-white/30">
          10m Candidate Comparison
        </p>

        <h2 className="mt-2 text-xl font-semibold">
          Frozen V1 vs RSI/Bollinger V2
        </h2>

        <p className="mt-2 max-w-3xl text-sm leading-6 text-white/40">
          V2 uses AI confidence ≥0.55, RSI 25/75 and a 20-bar Bollinger mean-reversion gate. It remains paper-only until fresh forward evidence supports it.
        </p>
      </div>

      {error && (
        <p className="mt-4 text-sm text-amber-200">
          {error}
        </p>
      )}

      {data && (
        <>
          <div className="mt-6 grid gap-4 md:grid-cols-2">
            <CandidateMetrics
              title="V1 · AI baseline"
              data={data.v1}
            />

            <CandidateMetrics
              title="V2 · AI + RSI + Bollinger"
              data={data.v2}
            />
          </div>

          <p className="mt-4 text-xs leading-5 text-white/30">
            {data.note}
          </p>
        </>
      )}
    </section>
  )
}


export default TenMinuteV2ComparisonPanel
