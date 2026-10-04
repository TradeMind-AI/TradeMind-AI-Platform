function Header() {
  return (
    <header className="border-b border-slate-800 bg-slate-900">

      <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-4">

        <div>

          <h2 className="text-xl font-bold">
            TradeMind AI
          </h2>

          <p className="text-xs text-slate-400">
            AI Trading Platform
          </p>

        </div>

        <div className="flex items-center gap-2">

          <div className="h-2.5 w-2.5 rounded-full bg-emerald-500" />

          <span className="text-sm text-slate-300">
            Paper Trading Active
          </span>

        </div>

      </div>

    </header>
  );
}

export default Header;