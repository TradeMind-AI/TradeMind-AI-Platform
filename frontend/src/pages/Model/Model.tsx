import {
  useEffect,
  useState,
} from 'react'

import DashboardPageShell
  from '../../components/layout/DashboardPageShell'

import StrategyResearch
  from '../../components/dashboard/StrategyResearch'

import SystemStatus
  from '../../components/dashboard/SystemStatus'

import {
  getDashboardData,
  getTimeframeStatus,
} from '../../services/tradingApi'

import type {
  StrategyTimeframe,
  TimeframeStatus,
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


const ModelPage = () => {
  const [
    timeframe,
    setTimeframe,
  ] = useState<StrategyTimeframe>('10m')

  const [
    statuses,
    setStatuses,
  ] = useState<TimeframeStatus[]>([])

  const [
    data,
    setData,
  ] = useState<TradingDashboardData | null>(null)

  const [
    error,
    setError,
  ] = useState<string | null>(null)


  useEffect(() => {
    getTimeframeStatus()
      .then(setStatuses)
      .catch(() => setStatuses([]))
  }, [])


  useEffect(() => {
    let active = true

    const load = async () => {
      try {
        const next =
          await getDashboardData(timeframe)

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
              : 'Model information unavailable',
          )
        }
      }
    }

    load()

    const timer =
      window.setInterval(load, 30_000)

    return () => {
      active = false
      window.clearInterval(timer)
    }
  }, [timeframe])


  const selectedStatus =
    statuses.find(
      (item) =>
        item.timeframe === timeframe,
    )


  return (
    <DashboardPageShell
      eyebrow="Model"
      title="Model & Research Registry"
      description="Inspect which strategy timeframes are ready, the active model metadata, feature count, confidence gate, forecast horizon and forward-research status."
    >
      <div className="flex flex-wrap gap-2">
        {TIMEFRAMES.map(
          (item) => {
            const status =
              statuses.find(
                (entry) =>
                  entry.timeframe === item,
              )

            return (
              <button
                key={item}
                type="button"
                onClick={() =>
                  setTimeframe(item)
                }
                className={[
                  'rounded-lg border px-4 py-2 text-sm transition',
                  timeframe === item
                    ? 'border-white/30 bg-white text-black'
                    : 'border-white/10 bg-white/[0.03] text-white/55 hover:text-white',
                ].join(' ')}
              >
                {item}

                {status
                  && !status.tradingReady && (
                    <span className="ml-2 text-[10px] opacity-50">
                      setup
                    </span>
                  )}
              </button>
            )
          },
        )}
      </div>

      {selectedStatus && (
        <div className="mt-6 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
          <ReadinessCard
            label="Market data"
            ready={selectedStatus.marketReady}
          />

          <ReadinessCard
            label="Features"
            ready={selectedStatus.featuresReady}
          />

          <ReadinessCard
            label="Model bundle"
            ready={selectedStatus.modelReady}
          />

          <ReadinessCard
            label="Paper log"
            ready={selectedStatus.paperLogReady}
          />
        </div>
      )}

      {error && (
        <div className="mt-6 rounded-2xl border border-amber-400/20 bg-amber-400/[0.05] p-6">
          <p className="font-medium text-amber-200">
            {timeframe} model is not fully available
          </p>

          <p className="mt-2 text-sm text-white/45">
            {error}
          </p>
        </div>
      )}

      {data && (
        <>
          <div className="mt-6 grid gap-6 xl:grid-cols-[1.2fr_0.8fr]">
            <StrategyResearch
              data={data}
            />

            <SystemStatus
              modelName={data.modelName}
              featureCount={data.featureCount}
              confidenceThreshold={
                data.confidenceThreshold
              }
              systemStatus={
                data.systemStatus
              }
            />
          </div>

          <section className="mt-6 rounded-2xl border border-white/10 bg-white/[0.025] p-6">
            <p className="text-xs uppercase tracking-[0.2em] text-white/30">
              Model interpretation
            </p>

            <div className="mt-5 grid gap-4 md:grid-cols-3">
              <Info
                label="Current prediction"
                value={data.prediction}
                note={
                  'Signal: '
                  + data.signal
                }
              />

              <Info
                label="Current confidence"
                value={
                  (
                    data.confidence
                    * 100
                  ).toFixed(2)
                  + '%'
                }
                note={
                  'Gate: '
                  + (
                    data.confidenceThreshold
                    * 100
                  ).toFixed(0)
                  + '%'
                }
              />

              <Info
                label="Evidence"
                value={
                  String(
                    data.closedForwardTrades,
                  )
                  + ' closed forward trades'
                }
                note={data.evidenceNote}
              />
            </div>
          </section>
        </>
      )}
    </DashboardPageShell>
  )
}


const ReadinessCard = ({
  label,
  ready,
}: {
  label: string
  ready: boolean
}) => (
  <div className="rounded-xl border border-white/10 bg-white/[0.025] p-4">
    <p className="text-xs uppercase tracking-wider text-white/25">
      {label}
    </p>

    <p
      className={[
        'mt-2 font-medium',
        ready
          ? 'text-emerald-300'
          : 'text-amber-300',
      ].join(' ')}
    >
      {ready ? 'Ready' : 'Pending'}
    </p>
  </div>
)


const Info = ({
  label,
  value,
  note,
}: {
  label: string
  value: string
  note: string
}) => (
  <div className="rounded-xl border border-white/5 bg-black/10 p-4">
    <p className="text-xs text-white/30">
      {label}
    </p>

    <p className="mt-2 text-lg font-medium">
      {value}
    </p>

    <p className="mt-2 text-xs leading-5 text-white/30">
      {note}
    </p>
  </div>
)


export default ModelPage
