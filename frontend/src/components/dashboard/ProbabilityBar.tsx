type ProbabilityBarProps = {
  label: string
  value: number
}

const ProbabilityBar = ({
  label,
  value,
}: ProbabilityBarProps) => {
  return (
    <div>
      <div className="flex justify-between text-sm">
        <span className="text-white/40">
          {label}
        </span>

        <span className="text-white/70">
          {(value * 100).toFixed(2)}%
        </span>
      </div>

      <div className="mt-2 h-2 overflow-hidden rounded-full bg-white/5">
        <div
          className="h-full rounded-full bg-white/60"
          style={{
            width: `${value * 100}%`,
          }}
        />
      </div>
    </div>
  )
}

export default ProbabilityBar