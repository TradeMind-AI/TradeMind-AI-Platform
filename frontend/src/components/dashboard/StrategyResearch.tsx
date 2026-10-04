import type {
  TradingDashboardData,
} from '../../types/trading'


type StrategyResearchProps = {
  data: TradingDashboardData
}


const Metric = ({
  label,
  value,
  subtext,
}: {
  label: string
  value: string
  subtext?: string
}) => {
  return (
    <div className="rounded-xl border border-white/5 bg-black/10 p-4">
      <p className="text-xs uppercase tracking-[0.16em] text-white/25">
        {label}
      </p>

      <p className="mt-2 text-lg font-medium text-white/80">
        {value}
      </p>

      {subtext && (
        <p className="mt-1 text-xs leading-5 text-white/30">
          {subtext}
        </p>
      )}
    </div>
  )
}


const StrategyResearch = ({
  data,
}: StrategyResearchProps) => {
  const isForward =
    data.strategyMode
    === 'Forward research'

  return (
    <section className="rounded-2xl border border-white/10 bg-white/[0.025] p-6">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <p className="text-xs uppercase tracking-[0.2em] text-white/30">
            Strategy Research
          </p>

          <h2 className="mt-2 text-xl font-semibold">
            {data.strategyMode}
          </h2>

          <p className="mt-2 max-w-3xl text-sm leading-6 text-white/40">
            {data.evidenceNote}
          </p>
        </div>

        <span
          className={[
            'w-fit rounded-full border px-3 py-1.5 text-xs font-medium',
            isForward
              ? 'border-blue-400/20 bg-blue-400/[0.08] text-blue-200'
              : 'border-white/10 bg-white/[0.04] text-white/50',
          ].join(' ')}
        >
          {data.researchStatus}
        </span>
      </div>

      <div className="mt-6 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <Metric
          label="Forecast horizon"
          value={`${data.forecastHorizonMinutes} min`}
          subtext="Fixed model forecast window"
        />

        <Metric
          label="Target move"
          value={`±${data.targetThresholdPct.toFixed(2)}%`}
          subtext="Directional training threshold"
        />

        <Metric
          label="Primary cost"
          value={`${data.primaryCostPct.toFixed(2)}%`}
          subtext="All-in round-trip research assumption"
        />

        <Metric
          label="Stress cost"
          value={`${data.stressCostPct.toFixed(2)}%`}
          subtext="Higher-cost robustness scenario"
        />
      </div>

      <div className="mt-4 grid gap-3 md:grid-cols-2">
        <div className="rounded-xl border border-white/5 bg-black/10 p-4">
          <p className="text-xs text-white/30">
            Execution model
          </p>

          <div className="mt-3 space-y-2 text-sm text-white/55">
            <p>
              Entry: <span className="text-white/80">{data.entryModel}</span>
            </p>

            <p>
              Exit: <span className="text-white/80">{data.exitModel}</span>
            </p>
          </div>
        </div>

        <div className="rounded-xl border border-white/5 bg-black/10 p-4">
          <p className="text-xs text-white/30">
            Forward evidence
          </p>

          <div className="mt-3 grid grid-cols-2 gap-4">
            <div>
              <p className="text-2xl font-semibold">
                {data.forwardSignalCount}
              </p>

              <p className="mt-1 text-xs text-white/30">
                decisions recorded
              </p>
            </div>

            <div>
              <p className="text-2xl font-semibold">
                {data.closedForwardTrades}
              </p>

              <p className="mt-1 text-xs text-white/30">
                closed trades
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}


export default StrategyResearch
