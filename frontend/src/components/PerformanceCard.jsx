function PerformanceCard({ data }) {
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900 p-6">

      <h3 className="text-lg font-semibold">
        System Status
      </h3>

      <div className="mt-6 space-y-5">

        <Row
          label="Model"
          value="Locked XGBoost"
        />

        <Row
          label="Features"
          value="84"
        />

        <Row
          label="Confidence Threshold"
          value="0.55"
        />

        <Row
          label="Latest Candle"
          value={data.latestCandle}
        />

        <Row
          label="Mode"
          value="Paper Trading"
        />

      </div>

    </div>
  );
}


function Row({
  label,
  value,
}) {
  return (
    <div className="flex items-center justify-between border-b border-slate-800 pb-3">

      <span className="text-sm text-slate-400">
        {label}
      </span>

      <span className="text-sm font-medium">
        {value}
      </span>

    </div>
  );
}


export default PerformanceCard;