function SignalCard({ data }) {
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900 p-6">

      <h3 className="text-lg font-semibold">
        Current Signal
      </h3>

      <div className="mt-6">

        <p className="text-sm text-slate-400">
          Signal
        </p>

        <p className="mt-1 text-4xl font-bold text-yellow-400">
          {data.signal}
        </p>

      </div>

      <div className="mt-6 grid grid-cols-2 gap-4">

        <div>

          <p className="text-sm text-slate-400">
            Prediction
          </p>

          <p className="mt-1 font-medium">
            {data.prediction}
          </p>

        </div>

        <div>

          <p className="text-sm text-slate-400">
            Confidence
          </p>

          <p className="mt-1 font-medium">
            {(data.confidence * 100).toFixed(2)}%
          </p>

        </div>

      </div>

      <div className="mt-6">

        <p className="mb-3 text-sm text-slate-400">
          Model Probabilities
        </p>

        <div className="space-y-3">

          <ProbabilityRow
            label="DOWN"
            value={data.probabilities.down}
          />

          <ProbabilityRow
            label="NEUTRAL"
            value={data.probabilities.neutral}
          />

          <ProbabilityRow
            label="UP"
            value={data.probabilities.up}
          />

        </div>

      </div>

    </div>
  );
}


function ProbabilityRow({
  label,
  value,
}) {
  return (
    <div>

      <div className="flex justify-between text-sm">

        <span>
          {label}
        </span>

        <span>
          {(value * 100).toFixed(2)}%
        </span>

      </div>

      <div className="mt-1 h-2 rounded-full bg-slate-800">

        <div
          className="h-2 rounded-full bg-blue-500"
          style={{
            width: `${value * 100}%`,
          }}
        />

      </div>

    </div>
  );
}


export default SignalCard;