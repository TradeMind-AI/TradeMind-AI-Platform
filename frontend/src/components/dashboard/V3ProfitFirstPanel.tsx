import {
  useEffect,
  useState,
} from 'react'

import {
  getV3ResearchSuite,
} from '../../services/v3ResearchApi'

import type {
  V3ResearchSuite,
  V3ResearchTrack,
} from '../../types/trading'


const TRACK_LABELS:
  Record<V3ResearchTrack['kind'], string> = {
    multitimeframe:
      'Multi-timeframe direction + profit selector',
    barrier:
      'Path-aware TP / SL opportunity model',
    transformer:
      'Transformer trade-success model',
  }


const ResearchTrackCard = ({
  track,
}: {
  track: V3ResearchTrack
}) => {
  const test = track.test

  return (
    <div className="rounded-xl border border-white/5 bg-black/10 p-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="text-sm font-medium text-white/80">
            {TRACK_LABELS[track.kind]}
          </p>

          <p className="mt-1 max-w-xl text-xs leading-5 text-white/30">
            {track.architecture ?? 'Research output unavailable'}
          </p>
        </div>

        <span
          className={[
            'rounded-full border px-2.5 py-1 text-[11px]',
            test?.targetMet
              ? 'border-emerald-400/20 bg-emerald-400/[0.08] text-emerald-200'
              : 'border-white/10 bg-white/[0.03] text-white/40',
          ].join(' ')}
        >
          {test?.targetMet
            ? 'Target met'
            : 'Rejected'}
        </span>
      </div>

      {!track.available || !test ? (
        <p className="mt-4 text-sm text-white/30">
          No completed result.
        </p>
      ) : (
        <div className="mt-4 grid grid-cols-2 gap-3 text-sm lg:grid-cols-5">
          <div>
            <p className="text-xs text-white/30">
              Test trades
            </p>
            <p className="mt-1 font-semibold">
              {test.trades}
            </p>
          </div>

          <div>
            <p className="text-xs text-white/30">
              Trades / day
            </p>
            <p className="mt-1 font-semibold">
              {test.tradesPerDay.toFixed(2)}
            </p>
          </div>

          <div>
            <p className="text-xs text-white/30">
              Win rate
            </p>
            <p className="mt-1 font-semibold">
              {test.winRatePct.toFixed(2)}%
            </p>
          </div>

          <div>
            <p className="text-xs text-white/30">
              Mid-cost net
            </p>
            <p className="mt-1 font-semibold">
              {test.netMidPct.toFixed(2)}%
            </p>
          </div>

          <div>
            <p className="text-xs text-white/30">
              High-cost net
            </p>
            <p className="mt-1 font-semibold">
              {test.netHighPct.toFixed(2)}%
            </p>
          </div>
        </div>
      )}
    </div>
  )
}


const V3ProfitFirstPanel = () => {
  const [
    data,
    setData,
  ] =
    useState<V3ResearchSuite | null>(
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
          await getV3ResearchSuite()

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

  const qualified =
    data?.promotionStatus
    === 'QUALIFIED_FOR_FORWARD_PAPER'

  return (
    <section className="rounded-2xl border border-white/10 bg-white/[0.025] p-6">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <p className="text-xs uppercase tracking-[0.2em] text-white/30">
            TradeMind V3 Research Suite
          </p>

          <h2 className="mt-2 text-xl font-semibold">
            Frequency + win rate + profit gate
          </h2>

          <p className="mt-2 max-w-3xl text-sm leading-6 text-white/40">
            A model is promoted only when validation and later-test evidence both satisfy the confidence, win-rate, trade-frequency, sample-size and transaction-cost requirements.
          </p>
        </div>

        <span
          className={[
            'w-fit rounded-full border px-3 py-1.5 text-xs font-medium',
            qualified
              ? 'border-emerald-400/20 bg-emerald-400/[0.08] text-emerald-200'
              : 'border-amber-400/20 bg-amber-400/[0.08] text-amber-200',
          ].join(' ')}
        >
          {qualified
            ? 'Eligible for forward paper'
            : data?.promotionStatus === 'NO_QUALIFIED_CANDIDATE'
              ? 'No candidate qualified'
              : 'Gate not evaluated'}
        </span>
      </div>

      {error && (
        <p className="mt-4 text-sm text-amber-200">
          {error}
        </p>
      )}

      {data && (
        <>
          <div className="mt-6 grid gap-3 sm:grid-cols-2 xl:grid-cols-5">
            <div className="rounded-xl border border-white/5 bg-black/10 p-4">
              <p className="text-xs text-white/30">
                Required confidence
              </p>
              <p className="mt-2 text-lg font-semibold">
                {(
                  Number(
                    data.requirements.minimum_confidence ?? 0,
                  ) * 100
                ).toFixed(0)}%
              </p>
            </div>

            <div className="rounded-xl border border-white/5 bg-black/10 p-4">
              <p className="text-xs text-white/30">
                Required win rate
              </p>
              <p className="mt-2 text-lg font-semibold">
                {Number(
                  data.requirements.minimum_win_rate_pct ?? 0,
                ).toFixed(0)}%
              </p>
            </div>

            <div className="rounded-xl border border-white/5 bg-black/10 p-4">
              <p className="text-xs text-white/30">
                Minimum frequency
              </p>
              <p className="mt-2 text-lg font-semibold">
                {Number(
                  data.requirements.minimum_trades_per_day ?? 0,
                ).toFixed(1)} / day
              </p>
            </div>

            <div className="rounded-xl border border-white/5 bg-black/10 p-4">
              <p className="text-xs text-white/30">
                Minimum sample
              </p>
              <p className="mt-2 text-lg font-semibold">
                {Number(
                  data.requirements.minimum_trades ?? 0,
                )} trades
              </p>
            </div>

            <div className="rounded-xl border border-white/5 bg-black/10 p-4">
              <p className="text-xs text-white/30">
                Qualified candidates
              </p>
              <p className="mt-2 text-lg font-semibold">
                {data.qualifiedCandidateCount}
              </p>
            </div>
          </div>

          <div className="mt-4 space-y-3">
            {data.tracks.map(
              (track) => (
                <ResearchTrackCard
                  key={track.kind}
                  track={track}
                />
              ),
            )}
          </div>

          <div className="mt-4 rounded-xl border border-white/5 bg-black/10 p-4">
            <p className="text-xs uppercase tracking-[0.14em] text-white/25">
              Promotion decision
            </p>

            <p className="mt-2 text-sm leading-6 text-white/55">
              {data.note}
            </p>
          </div>
        </>
      )}
    </section>
  )
}


export default V3ProfitFirstPanel
