import { Link } from 'react-router-dom'

const Landing = () => {
  return (
    <main className="min-h-screen bg-[#050505] text-[#f6f0df]">
      {/* Navigation */}
      <nav className="mx-auto flex max-w-7xl items-center justify-between px-6 py-6 lg:px-8">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl border border-[#d4af37]/35 bg-[#d4af37]/[0.06] shadow-[0_0_26px_rgba(212,175,55,0.08)]">
            <span className="text-lg font-bold text-[#f0cf68]">T</span>
          </div>

          <span className="text-xl font-semibold tracking-tight">
            TradeMind <span className="text-[#d4af37]">AI</span>
          </span>
        </div>

        <div className="hidden items-center gap-8 text-sm text-white/60 md:flex">
          <a href="#platform" className="transition hover:text-white">
            Platform
          </a>

          <a href="#intelligence" className="transition hover:text-white">
            Intelligence
          </a>

          <a href="#trading" className="transition hover:text-white">
            Trading
          </a>
        </div>

        <div className="flex items-center gap-3">
          <button className="hidden px-4 py-2 text-sm text-white/70 transition hover:text-white sm:block">
            Log in
          </button>

          <button className="rounded-lg border border-[#d4af37]/50 bg-[linear-gradient(135deg,#f0cf68,#d4af37)] px-5 py-2.5 text-sm font-semibold text-black shadow-[0_10px_30px_rgba(212,175,55,0.12)] transition hover:brightness-110">
            Get Started
          </button>
        </div>
      </nav>

      {/* Hero */}
      <section className="relative overflow-hidden">
        {/* Background glow */}
        <div className="pointer-events-none absolute left-1/2 top-20 h-96 w-96 -translate-x-1/2 rounded-full bg-[#d4af37]/10 blur-3xl" />

        <div className="relative mx-auto flex min-h-[calc(100vh-88px)] max-w-7xl flex-col items-center justify-center px-6 py-24 text-center lg:px-8">
          
          {/* Small label */}
          <div className="mb-8 rounded-full border border-white/10 bg-white/[0.03] px-4 py-2 text-xs font-medium uppercase tracking-[0.2em] text-white/50">
            AI-powered trading intelligence
          </div>

          {/* Main heading */}
          <h1 className="max-w-5xl text-5xl font-semibold leading-[1.05] tracking-tight sm:text-6xl lg:text-8xl">
            Trade with
            <span className="block text-[#c5a64a]">
              intelligence.
            </span>
          </h1>

          {/* Description */}
          <p className="mt-8 max-w-2xl text-base leading-7 text-white/50 sm:text-lg">
            TradeMind AI combines market data, machine learning, and
            cost-aware validation to analyze Bitcoin markets and support
            systematic paper-trading research.
          </p>

          {/* Buttons */}
          <div className="mt-10 flex flex-col gap-4 sm:flex-row">
            <button className="rounded-xl border border-[#d4af37]/50 bg-[linear-gradient(135deg,#f0cf68,#d4af37)] px-7 py-3.5 text-sm font-semibold text-black shadow-[0_12px_34px_rgba(212,175,55,0.14)] transition hover:brightness-110">
              Start Paper Trading
            </button>

            <Link
              to="/dashboard"
              className="rounded-xl border border-[#d4af37]/20 bg-[#d4af37]/[0.035] px-7 py-3.5 text-sm font-medium text-[#e8d9ad] transition hover:border-[#d4af37]/35 hover:bg-[#d4af37]/[0.07] hover:text-[#f0cf68]"
            >
              Explore the Platform
            </Link>
          </div>

          {/* Trust statement */}
          <p className="mt-8 text-xs text-white/30">
            Start in paper trading mode. No real funds required.
          </p>

          {/* Dashboard Preview */}
          <div className="mt-20 w-full max-w-6xl">
            <div className="rounded-2xl border border-white/10 bg-white/[0.025] p-2 shadow-2xl shadow-black/40">
              <div className="rounded-xl border border-white/5 bg-[#080d18] p-5">
                
                {/* Fake dashboard header */}
                <div className="flex flex-col justify-between gap-4 border-b border-white/5 pb-5 sm:flex-row sm:items-center">
                  <div>
                    <p className="text-left text-xs uppercase tracking-wider text-white/30">
                      Market
                    </p>

                    <div className="mt-1 flex items-center gap-3">
                      <span className="text-xl font-semibold">
                        BTC / USDT
                      </span>

                      <span className="rounded-md bg-white/5 px-2 py-1 text-xs text-white/50">
                        1H
                      </span>
                    </div>
                  </div>

                  <div className="text-left sm:text-right">
                    <p className="text-xs text-white/30">
                      AI confidence
                    </p>

                    <p className="mt-1 text-lg font-medium">
                      76.4%
                    </p>
                  </div>
                </div>

                {/* Fake chart */}
                <div className="relative mt-6 h-64 overflow-hidden rounded-xl bg-[#050914]">
                  <div className="absolute inset-0 opacity-20">
                    <div className="h-full w-full bg-[linear-gradient(rgba(255,255,255,0.08)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.08)_1px,transparent_1px)] bg-[size:48px_48px]" />
                  </div>

                  <svg
                    viewBox="0 0 1000 300"
                    className="absolute inset-0 h-full w-full"
                    preserveAspectRatio="none"
                  >
                    <path
                      d="M0 240 C80 230 100 250 170 210 C240 170 270 190 330 145 C390 100 420 150 470 120 C530 85 550 110 610 80 C680 45 700 100 760 65 C830 25 870 70 930 40 C960 25 980 35 1000 20"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="3"
                      className="text-[#d4af37]"
                    />
                  </svg>

                  <div className="absolute bottom-5 left-5 rounded-lg border border-white/10 bg-black/30 px-4 py-3 backdrop-blur">
                    <p className="text-xs text-white/30">
                      AI Signal
                    </p>

                    <p className="mt-1 text-sm font-semibold">
                      BUY
                    </p>
                  </div>
                </div>

                {/* Dashboard cards */}
                <div className="mt-4 grid grid-cols-2 gap-4 md:grid-cols-4">
                  {[
                    ["Market Trend", "Bullish"],
                    ["AI Signal", "BUY"],
                    ["Risk Level", "Moderate"],
                    ["Mode", "Paper"],
                  ].map(([label, value]) => (
                    <div
                      key={label}
                      className="rounded-xl border border-white/5 bg-white/[0.02] p-4 text-left"
                    >
                      <p className="text-xs text-white/30">
                        {label}
                      </p>

                      <p className="mt-2 text-sm font-medium text-white/80">
                        {value}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Intelligence section */}
      <section
        id="intelligence"
        className="border-t border-white/5 px-6 py-28 lg:px-8"
      >
        <div className="mx-auto max-w-7xl">
          <div className="max-w-2xl">
            <p className="text-xs font-medium uppercase tracking-[0.2em] text-white/30">
              Built for intelligent decisions
            </p>

            <h2 className="mt-5 text-4xl font-semibold tracking-tight sm:text-5xl">
              More than a trading signal.
            </h2>

            <p className="mt-6 text-base leading-7 text-white/45">
              TradeMind AI is designed to combine multiple sources of
              information instead of relying on a single indicator or model.
            </p>
          </div>

          <div className="mt-14 grid gap-4 md:grid-cols-3">
            {[
              {
                title: "Machine Learning",
                description:
                  "Models learn patterns from historical Bitcoin market data and engineered features.",
              },
              {
                title: "Multi-Timeframe Research",
                description:
                  "Independent models and validation pipelines compare signals across multiple market horizons.",
              },
              {
                title: "Risk-Aware Execution",
                description:
                  "Signals pass through risk management before a trading decision is executed.",
              },
            ].map((item) => (
              <div
                key={item.title}
                className="rounded-2xl border border-white/10 bg-white/[0.025] p-7"
              >
                <h3 className="text-lg font-medium">
                  {item.title}
                </h3>

                <p className="mt-4 text-sm leading-6 text-white/40">
                  {item.description}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-white/5 px-6 py-8 lg:px-8">
        <div className="mx-auto flex max-w-7xl flex-col justify-between gap-4 text-sm text-white/30 sm:flex-row">
          <p>© 2026 TradeMind AI</p>

          <p>
            AI-powered market intelligence & automated trading research
          </p>
        </div>
      </footer>
    </main>
  )
}

export default Landing