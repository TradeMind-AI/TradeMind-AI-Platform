import {
  useEffect,
  useState,
} from 'react'

import DashboardPageShell
  from '../../components/layout/DashboardPageShell'

import {
  getStagedForwardResearch,
} from '../../services/stagedForwardApi'

import type {
  StagedForwardResearch,
} from '../../types/trading'


const pct = (
  value: number | null,
  digits = 2,
) => (
  value === null
    ? '—'
    : value.toFixed(digits) + '%'
)


const ProgressBar = ({
  label,
  current,
  target,
  progress,
}: {
  label: string
  current: number
  target: number
  progress: number
}) => (
  <div>
    <div className="flex items-center justify-between gap-4 text-sm">
      <span className="text-white/45">
        {label}
      </span>

      <span className="text-white/70">
        {current.toLocaleString()}
        {' / '}
        {target.toLocaleString()}
      </span>
    </div>

    <div className="mt-2 h-2 overflow-hidden rounded-full bg-white/[0.06]">
      <div
        className="h-full rounded-full bg-white/50 transition-all duration-500"
        style={{
          width:
            Math.min(
              100,
              Math.max(0, progress),
            ) + '%',
        }}
      />
    </div>

    <p className="mt-2 text-xs text-white/25">
      {progress.toFixed(2)}% complete
    </p>
  </div>
)


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

    <p className="mt-1 text-xs leading-5 text-white/30">
      {note}
    </p>
  </div>
)


const FunnelStep = ({
  number,
  title,
  text,
}: {
  number: string
  title: string
  text: string
}) => (
  <div className="flex gap-4 rounded-xl border border-white/5 bg-black/10 p-4">
    <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg border border-white/10 bg-white/[0.04] text-sm font-semibold text-white/70">
      {number}
    </div>

    <div>
      <p className="font-medium text-white/80">
        {title}
      </p>

      <p className="mt-1 text-sm leading-6 text-white/35">
        {text}
      </p>
    </div>
  </div>
)


const ForwardValidationPage = () => {
  const [data, setData] =
    useState<StagedForwardResearch | null>(null)

  const [error, setError] =
    useState<string | null>(null)


  useEffect(() => {
    let active = true

    const load = async () => {
      try {
        const next =
          await getStagedForwardResearch()

        if (!active) {
          return
        }

        setData(next)
        setError(null)
      } catch (nextError) {
        if (!active) {
          return
        }

        setError(
          nextError instanceof Error
            ? nextError.message
            : 'Forward validation data is unavailable.',
        )
      }
    }

    load()

    const timer =
      window.setInterval(load, 15_000)

    return () => {
      active = false
      window.clearInterval(timer)
    }
  }, [])


  const verdictText = (
    data?.verdict
      ?.replaceAll('_', ' ')
      ?? 'LOADING'
  )

  return (
    <DashboardPageShell
      eyebrow="Forward validation"
      title="Frozen Model Evidence"
      description="Monitor genuinely new post-freeze evidence for the staged microstructure candidate. This page reports research results only and does not place real trades."
    >
      {error && (
        <div className="mb-6 rounded-xl border border-amber-400/20 bg-amber-400/[0.05] px-4 py-3 text-sm text-amber-200">
          {error}
        </div>
      )}

      <section className="rounded-2xl border border-white/10 bg-white/[0.025] p-6">
        <div className="flex flex-col gap-5 lg:flex-row lg:items-start lg:justify-between">
          <div>
            <p className="text-xs uppercase tracking-[0.2em] text-white/30">
              Research verdict
            </p>

            <h3 className="mt-2 text-2xl font-semibold">
              {verdictText}
            </h3>

            <p className="mt-3 max-w-3xl text-sm leading-6 text-white/35">
              {data?.verdictNote
                ?? 'Loading frozen-candidate evidence...'}
            </p>
          </div>

          <div className="rounded-xl border border-white/10 bg-black/10 px-4 py-3 text-sm text-white/45">
            <p>
              Model: {data?.modelName ?? '—'}
            </p>

            <p className="mt-1">
              Status: {data?.evidenceStatus ?? '—'}
            </p>
          </div>
        </div>
      </section>

      <div className="mt-6 grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <MetricCard
          label="New snapshots"
          value={
            data
              ? data.newSnapshots.toLocaleString()
              : '—'
          }
          note="Post-freeze observations only"
        />

        <MetricCard
          label="Evaluable rows"
          value={
            data
              ? data.evaluableForwardRows.toLocaleString()
              : '—'
          }
          note="Rows with a complete 60-second future path"
        />

        <MetricCard
          label="Selected trades"
          value={
            data
              ? data.selectedPredictions.toLocaleString()
              : '—'
          }
          note={
            data
              ? pct(data.coveragePct, 3)
                + ' forward coverage'
              : 'Frozen selection policy'
          }
        />

        <MetricCard
          label="Qualified trade rate"
          value={
            data
              ? pct(
                  data.qualifiedTradeRatePct,
                  3,
                )
              : '—'
          }
          note={
            data
              ? 'Target ≥ '
                + data.minimumNetProfitPct.toFixed(3)
                + '% net after modeled fees'
              : 'Cost-aware success rate'
          }
        />
      </div>

      <div className="mt-6 grid gap-6 xl:grid-cols-2">
        <section className="rounded-2xl border border-white/10 bg-white/[0.025] p-6">
          <p className="text-xs uppercase tracking-[0.2em] text-white/30">
            Forward evidence
          </p>

          <div className="mt-5 space-y-4">
            <div className="flex items-center justify-between gap-4 border-b border-white/5 pb-3 text-sm">
              <span className="text-white/35">
                Frozen at
              </span>

              <span className="text-right text-white/70">
                {data?.frozenAt
                  ? new Date(
                      data.frozenAt,
                    ).toLocaleString()
                  : '—'}
              </span>
            </div>

            <div className="flex items-center justify-between gap-4 border-b border-white/5 pb-3 text-sm">
              <span className="text-white/35">
                Average selected net return
              </span>

              <span className="text-right text-white/70">
                {data
                  ? pct(
                      data.avgSelectedNetReturnPct,
                      4,
                    )
                  : '—'}
              </span>
            </div>

            <div className="flex items-center justify-between gap-4 text-sm">
              <span className="text-white/35">
                Deployment status
              </span>

              <span className="text-right font-medium text-white/80">
                Research only
              </span>
            </div>
          </div>

          {data
            && data.selectedPredictions < 20
            && (
              <div className="mt-5 rounded-xl border border-amber-300/15 bg-amber-300/[0.04] p-4 text-sm leading-6 text-amber-100/70">
                Only {data.selectedPredictions} selected forward trades are available. Percentages at this sample size can change sharply and should not be interpreted as proof of profitability.
              </div>
            )}
        </section>

        <section className="rounded-2xl border border-white/10 bg-white/[0.025] p-6">
          <p className="text-xs uppercase tracking-[0.2em] text-white/30">
            Collection checkpoints
          </p>

          <div className="mt-5 space-y-7">
            <ProgressBar
              label="Minimum research checkpoint"
              current={data?.newSnapshots ?? 0}
              target={data?.minimumCheckpoint ?? 25_000}
              progress={
                data?.minimumCheckpointProgressPct
                ?? 0
              }
            />

            <ProgressBar
              label="Preferred research checkpoint"
              current={data?.newSnapshots ?? 0}
              target={data?.preferredCheckpoint ?? 50_000}
              progress={
                data?.preferredCheckpointProgressPct
                ?? 0
              }
            />
          </div>
        </section>
      </div>

      <section className="mt-6 rounded-2xl border border-white/10 bg-white/[0.025] p-6">
        <p className="text-xs uppercase tracking-[0.2em] text-white/30">
          Staged tradeability funnel
        </p>

        <div className="mt-5 grid gap-3 lg:grid-cols-3">
          <FunnelStep
            number="1"
            title="Trade or skip"
            text="Microstructure features first estimate whether the setup is worth considering after modeled costs."
          />

          <FunnelStep
            number="2"
            title="Long or short"
            text="Only candidate setups continue to the directional model, which chooses the proposed side."
          />

          <FunnelStep
            number="3"
            title="Profitability filter"
            text="The final selector estimates whether the proposed side can clear the frozen minimum net-profit requirement."
          />
        </div>
      </section>

      <section className="mt-6 rounded-2xl border border-white/10 bg-white/[0.025] p-6">
        <p className="text-xs uppercase tracking-[0.2em] text-white/30">
          Research guardrails
        </p>

        <p className="mt-3 max-w-4xl text-sm leading-6 text-white/35">
          The candidate is frozen so post-freeze observations remain genuine forward evidence. Results shown here are paper-research metrics, not financial advice, not live execution results, and not a guarantee of future performance.
        </p>
      </section>
    </DashboardPageShell>
  )
}


export default ForwardValidationPage
