import ProbabilityBar from './ProbabilityBar'
import type {
  Prediction,
  ProbabilitySet,
  TradingSignal,
} from '../../types/trading'

type SignalCardProps = {
  prediction: Prediction
  confidence: number
  signal: TradingSignal
  probabilities: ProbabilitySet
}

const getSignalClass = (
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

const SignalCard = ({
  prediction,
  confidence,
  signal,
  probabilities,
}: SignalCardProps) => {
  return (
    <section className="rounded-2xl border border-white/10 bg-white/[0.025] p-6">
      <p className="text-xs uppercase tracking-[0.2em] text-white/30">
        Current AI Signal
      </p>

      <div className="mt-6 flex items-end justify-between gap-4">
        <div>
          <p
            className={[
              'text-4xl font-semibold',
              getSignalClass(signal),
            ].join(' ')}
          >
            {signal}
          </p>

          <p className="mt-2 text-sm text-white/40">
            Prediction: {prediction}
          </p>
        </div>

        <div className="text-right">
          <p className="text-xs text-white/30">
            Confidence
          </p>

          <p className="mt-1 text-2xl font-medium">
            {(confidence * 100).toFixed(2)}%
          </p>
        </div>
      </div>

      <div className="mt-8 space-y-4">
        <ProbabilityBar
          label="DOWN"
          value={probabilities.down}
        />

        <ProbabilityBar
          label="NEUTRAL"
          value={probabilities.neutral}
        />

        <ProbabilityBar
          label="UP"
          value={probabilities.up}
        />
      </div>
    </section>
  )
}

export default SignalCard