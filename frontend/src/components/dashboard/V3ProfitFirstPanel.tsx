import {
  useEffect,
  useState,
} from 'react'

import {
  getV3ProfitFirstResearch,
} from '../../services/v3ResearchApi'

import type {
  V3ProfitFirstResearch,
} from '../../types/trading'


const Metric = ({
  label,
  value,
  subtext,
}: {
  label: string
  value: string
  subtext?: string
}) => (
  <div className="rounded-xl border border-white/5 bg-black/10 p-4">
    <p className="text-xs uppercase tracking-[0.14em] text-white/25">
      {label}
    </p>

    <p className="mt-2 text-lg font-semibold text-white/80">
      {value}
    </p>

    {subtext && (
      <p className="mt-1 text-xs leading-5 text-white/30">
        {subtext}
      </p>
    )}
  </div>
)


const V3ProfitFirstPanel = () => {
  const [
    data,
    setData,
  ] =
    useState<V3ProfitFirstResearch | null>(
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
          await getV3ProfitFirstResearch()

        if (active) {
          setData(next)
          setError(null)
        }
      } catch (err) {
        if (active) {
          setError(
            err instanceof Error
              ? err.message
              : 'Failed to load V3 research',
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

  const candidate =
    data?.bestCandidate ?? null

  return (
    <section className="rounded-2xl border border-white/10 bg-white/[0.025] p-6">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <p className="text-xs uppercase tracking-[0.2em] text-white/30">
            TradeMind V3 Research
          </p>

          <h2 className="mt-2 text-xl font-semibold">
            Profit-first · frequency-aware AI
          </h2>

          <p className="mt-2 max-w-3xl text-sm leading-6 text-white/40">
            V3 removes NEUTRAL from the direction proposer. It proposes LONG or SHORT, then a separate profitability model decides whether the trade is worth executing after costs.
          </p>
        </div>

        <span
          className={[
            'w-fit rounded-full border px-3 py-1.5 text-xs font-medium',
            data?.status === 'TARGET_MET'
              ? 'border-emerald-400/20 bg-emerald-400/[0.08] text-emerald-200'
              : 'border-amber-400/20 bg-amber-400/[0.08] text-amber-200',
          ].join(' ')}
        >
          {data?.status === 'TARGET_MET'
            ? 'Target met'
            : data?.available
              ? 'Research only'
              : 'Awaiting results'}
        </span>
      </div>

      {error && (
        <p className="mt-4 text-sm text-amber-200">
          {error}
        </p>
      )}

      {!error && data && !data.available && (
        <div className="mt-6 rounded-xl border border-white/5 bg-black/10 p-4 text-sm text-white/40">
          {data.message}
        </div>
      )}

      {candidate && (
        <>
          <div className="mt-6 rounded-xl border border-white/5 bg-black/10 p-4">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <p className="text-xs text-white/30">
                  Validation-selected candidate
                </p>

                <p className="mt-1 text-lg font-semibold">
                  {candidate.timeframe} · {candidate.horizonMinutes}m horizon
                </p>
              </div>

              <p className="text-xs text-white/30">
                proposer ≥ {(candidate.proposerThreshold * 100).toFixed(0)}% · selector ≥ {(candidate.selectorThreshold * 100).toFixed(0)}%
              </p>
            </div>
          </div>

          <div className="mt-4 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
            <Metric
              label="Later-test win rate"
              value={`${candidate.testWinRatePct.toFixed(2)}%`}
              subtext="Net-positive trades after the mid-cost assumption"
            />

            <Metric
              label="Later-test frequency"
              value={`${candidate.testTradesPerDay.toFixed(2)} / day`}
              subtext={`${candidate.testTrades} later-test trades`}
            />

            <Metric
              label="Later-test mid-cost net"
              value={`${candidate.testNetMidPct.toFixed(2)}%`}
              subtext="Summed research trade return"
            />

            <Metric
              label="High-cost stress net"
              value={`${candidate.testNetHighPct.toFixed(2)}%`}
              subtext="Must remain positive for strict target"
            />
          </div>

          <div className="mt-4 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
            <Metric
              label="Validation win rate"
              value={`${candidate.validationWinRatePct.toFixed(2)}%`}
            />

            <Metric
              label="Validation frequency"
              value={`${candidate.validationTradesPerDay.toFixed(2)} / day`}
            />

            <Metric
              label="Validation trades"
              value={String(candidate.validationTrades)}
            />

            <Metric
              label="Strict target"
              value={candidate.testTargetMet ? 'PASS' : 'NOT PROVEN'}
              subtext="80% win target + frequency + positive cost stress"
            />
          </div>

          <p className="mt-4 text-xs leading-5 text-white/30">
            {data?.message ?? ''}
          </p>
        </>
      )}
    </section>
  )
}


export default V3ProfitFirstPanel
