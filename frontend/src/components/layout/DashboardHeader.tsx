type DashboardHeaderProps = {
  symbol: string
  timeframe: string
  latestCandle: string
}

const DashboardHeader = ({
  symbol,
  timeframe,
  latestCandle,
}: DashboardHeaderProps) => {
  return (
    <header className="border-b border-white/5 bg-[#030712]/95 px-6 py-5 backdrop-blur lg:px-8">
      <div className="flex flex-col justify-between gap-4 md:flex-row md:items-center">
        <div>
          <p className="text-xs uppercase tracking-[0.2em] text-white/30">
            Live Market
          </p>

          <div className="mt-2 flex items-center gap-3">
            <h1 className="text-2xl font-semibold">
              {symbol}
            </h1>

            <span className="rounded-md bg-white/5 px-2 py-1 text-xs text-white/40">
              {timeframe}
            </span>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <div className="text-right">
            <p className="text-xs text-white/30">
              Latest closed candle
            </p>

            <p className="mt-1 text-sm text-white/70">
              {latestCandle}
            </p>
          </div>

          <div className="h-2.5 w-2.5 rounded-full bg-emerald-400" />
        </div>
      </div>
    </header>
  )
}

export default DashboardHeader