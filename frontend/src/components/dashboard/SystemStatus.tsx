type SystemStatusProps = {
  modelName: string
  featureCount: number
  confidenceThreshold: number
  systemStatus: string
}

const StatusRow = ({
  label,
  value,
}: {
  label: string
  value: string
}) => {
  return (
    <div className="flex items-center justify-between border-b border-white/5 pb-4">
      <span className="text-sm text-white/35">
        {label}
      </span>

      <span className="text-sm font-medium text-white/70">
        {value}
      </span>
    </div>
  )
}

const SystemStatus = ({
  modelName,
  featureCount,
  confidenceThreshold,
  systemStatus,
}: SystemStatusProps) => {
  return (
    <section className="rounded-2xl border border-white/10 bg-white/[0.025] p-6">
      <p className="text-xs uppercase tracking-[0.2em] text-white/30">
        System Status
      </p>

      <div className="mt-6 space-y-4">
        <StatusRow
          label="Model"
          value={modelName}
        />

        <StatusRow
          label="Features"
          value={String(featureCount)}
        />

        <StatusRow
          label="Confidence threshold"
          value={confidenceThreshold.toFixed(2)}
        />

        <StatusRow
          label="Trading mode"
          value="Paper Trading"
        />

        <StatusRow
          label="Status"
          value={systemStatus}
        />
      </div>
    </section>
  )
}

export default SystemStatus