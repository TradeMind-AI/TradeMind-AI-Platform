import type {
  RecentSignal,
  TradingSignal,
} from '../../types/trading'

type RecentSignalsProps = {
  signals: RecentSignal[]
}

const signalClass = (
  signal: TradingSignal,
) => {
  if (signal === 'BUY') {
    return 'text-emerald-400'
  }

  if (signal === 'SELL') {
    return 'text-red-400'
  }

  return 'text-amber-300'
}

const RecentSignals = ({
  signals,
}: RecentSignalsProps) => {
  return (
    <section className="overflow-hidden rounded-2xl border border-white/10 bg-white/[0.025]">
      <div className="border-b border-white/5 px-6 py-5">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-xs uppercase tracking-[0.2em] text-white/30">
              Activity
            </p>

            <h2 className="mt-2 text-lg font-semibold">
              Recent Signals
            </h2>
          </div>

          <span className="text-xs text-white/25">
            Latest model decisions
          </span>
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full min-w-[700px] text-left">
          <thead>
            <tr className="border-b border-white/5 text-xs uppercase tracking-wider text-white/25">
              <th className="px-6 py-4">
                Time
              </th>

              <th className="px-6 py-4">
                Prediction
              </th>

              <th className="px-6 py-4">
                Confidence
              </th>

              <th className="px-6 py-4">
                Signal
              </th>

              <th className="px-6 py-4">
                Status
              </th>
            </tr>
          </thead>

          <tbody>
            {signals.map((item) => (
              <tr
                key={item.timestamp}
                className="border-b border-white/5 last:border-b-0"
              >
                <td className="px-6 py-4 text-sm text-white/50">
                  {item.timestamp}
                </td>

                <td className="px-6 py-4 text-sm">
                  {item.prediction}
                </td>

                <td className="px-6 py-4 text-sm">
                  {(item.confidence * 100).toFixed(2)}%
                </td>

                <td
                  className={[
                    'px-6 py-4 text-sm font-semibold',
                    signalClass(item.signal),
                  ].join(' ')}
                >
                  {item.signal}
                </td>

                <td className="px-6 py-4 text-sm text-white/40">
                  {item.status}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}

export default RecentSignals