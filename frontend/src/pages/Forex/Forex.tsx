import { useCallback, useEffect, useRef, useState } from 'react'
import { createChart, CandlestickSeries, type UTCTimestamp } from 'lightweight-charts'
import Sidebar from '../../components/layout/Sidebar'
import {
  forexStart, forexStop, forexStep, getForexCandles, getForexStatus, updateForexSettings,
  type ForexCandle, type ForexState, type ForexTrade,
} from '../../services/forexApi'

const money = (value: number) => value.toLocaleString('en-US', { style: 'currency', currency: 'USD' })
const stamp = (value: number) => new Date(value * 1000).toLocaleString()
const panel = 'rounded-2xl border border-white/10 bg-[#0a1224] p-5'
const button = 'rounded-lg border border-white/15 px-3 py-2 text-sm text-white/80 hover:bg-white/10 disabled:opacity-40'

function CandleChart({ candles }: { candles: ForexCandle[] }) {
  const container = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (!container.current || !candles.length) return
    const chart = createChart(container.current, {
      width: container.current.clientWidth, height: 395,
      layout: { background: { color: '#0a1224' }, textColor: '#94a3b8' },
      grid: { vertLines: { color: '#182339' }, horzLines: { color: '#182339' } },
      rightPriceScale: { borderColor: '#25314a' },
      timeScale: { borderColor: '#25314a', timeVisible: true },
    })
    const series = chart.addSeries(CandlestickSeries, {
      upColor: '#34d399', downColor: '#fb7185',
      borderUpColor: '#34d399', borderDownColor: '#fb7185',
      wickUpColor: '#34d399', wickDownColor: '#fb7185',
      priceFormat: { type: 'price', precision: 5, minMove: 0.00001 },
    })
    series.setData(candles.map(c => ({ ...c, time: c.time as UTCTimestamp })))
    chart.timeScale().fitContent()
    const observer = new ResizeObserver(() => {
      if (container.current) chart.applyOptions({ width: container.current.clientWidth })
    })
    observer.observe(container.current)
    return () => { observer.disconnect(); chart.remove() }
  }, [candles])
  return <div ref={container} className="w-full" role="img" aria-label="Simulated EUR/USD candlestick chart" />
}

export default function Forex() {
  const [state, setState] = useState<ForexState | null>(null)
  const [candles, setCandles] = useState<ForexCandle[]>([])
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [filter, setFilter] = useState('ALL')
  const [draft, setDraft] = useState<ForexState['settings'] | null>(null)
  const ticking = useRef(false)

  const refresh = useCallback(async () => {
    try {
      const [s, data] = await Promise.all([getForexStatus(), getForexCandles()])
      setState(s); setCandles(data.candles); setError('')
    } catch (e) { setError(e instanceof Error ? e.message : 'Unable to reach Forex API') }
  }, [])

  useEffect(() => { void refresh() }, [refresh])
  useEffect(() => {
    if (!state) return
    setDraft(previous => previous ?? state.settings)
  }, [state])
  useEffect(() => {
    if (!state?.running) return
    const timer = window.setInterval(async () => {
      if (ticking.current) return
      ticking.current = true
      try { setState(await forexStep()); setCandles((await getForexCandles()).candles); setError('') }
      catch (e) { setError(e instanceof Error ? e.message : 'Simulation step failed') }
      finally { ticking.current = false }
    }, 2500)
    return () => window.clearInterval(timer)
  }, [state?.running])

  const toggle = async () => {
    setBusy(true)
    try { setState(await (state?.running ? forexStop() : forexStart())); setError('') }
    catch (e) { setError(e instanceof Error ? e.message : 'Could not change bot status') }
    finally { setBusy(false) }
  }
  const saveSettings = async () => {
    if (!draft) return
    setBusy(true)
    try { setState(await updateForexSettings(draft)); setError('') }
    catch (e) { setError(e instanceof Error ? e.message : 'Could not update settings') }
    finally { setBusy(false) }
  }
  const exportCsv = () => {
    if (!state) return
    const headers: (keyof ForexTrade)[] = ['id','entry_time','exit_time','side','entry','exit','units','stop','target','pnl','pnl_pct','result','confidence','reason','reason_exit']
    const csv = [headers.join(','), ...state.trades.map(t => headers.map(k => JSON.stringify(t[k] ?? '')).join(','))].join('\n')
    const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8' }))
    const anchor = document.createElement('a'); anchor.href = url; anchor.download = 'forex-paper-trades.csv'
    anchor.click(); URL.revokeObjectURL(url)
  }
  const trades = state?.trades.filter(t => filter === 'ALL' || t.result === filter || t.side === filter) ?? []

  return <div className="min-h-screen bg-[#030712] text-white"><div className="flex">
    <Sidebar />
    <main className="min-w-0 flex-1 px-5 py-7 lg:px-8">
      <div className="mx-auto max-w-[1500px] space-y-6">
        <header className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <p className="text-xs uppercase tracking-[0.23em] text-cyan-400">TradeMind AI / Forex Bot</p>
            <h1 className="mt-2 text-3xl font-semibold">EUR/USD AI Paper Trader</h1>
            <p className="mt-2 text-sm text-slate-400">Simulated 15-minute EUR/USD market · No real orders</p>
          </div>
          <span className="rounded-full border border-amber-500/40 bg-amber-500/10 px-4 py-2 text-xs font-bold tracking-wider text-amber-300">PAPER TRADING ONLY</span>
        </header>
        {error && <div role="alert" className="rounded-lg border border-rose-500/40 bg-rose-500/10 px-4 py-3 text-sm text-rose-300">{error} <button className="ml-4 underline" onClick={() => void refresh()}>Retry</button></div>}
        {!state ? <div className={panel}>Loading forex simulator from FastAPI…</div> : <>
          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            {[
              ['Paper balance', money(state.balance)],
              ['Closed trades', String(state.metrics.trades)],
              ['Net P&L', money(state.metrics.net_pnl)],
              ['Win rate', state.metrics.win_rate.toFixed(2) + '% · target 80%'],
            ].map(([name,value]) => <div key={name} className={panel}><p className="text-xs uppercase tracking-wider text-slate-400">{name}</p><p className="mt-3 text-2xl font-semibold">{value}</p></div>)}
          </div>
          <div className="grid gap-6 xl:grid-cols-[minmax(0,1.8fr)_minmax(310px,0.8fr)]">
            <section className={panel}>
              <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
                <div><h2 className="text-lg font-semibold">EUR / USD · 15m</h2><p className="text-xs text-amber-300">Deterministic simulated feed — not real market prices</p></div>
                <p className="font-mono text-xl">{candles.at(-1)?.close.toFixed(5) ?? '—'}</p>
              </div>
              <CandleChart candles={candles} />
              <p className="mt-2 text-xs text-slate-500">TradingView-style chart powered by Lightweight Charts. Demo timeline advances one simulated candle every 2.5 seconds while the page is open.</p>
            </section>
            <section className={panel}>
              <div className="flex items-center justify-between"><h2 className="text-lg font-semibold">Bot controls</h2><span className={state.running ? 'text-emerald-400' : 'text-slate-400'}>{state.running ? (state.position ? 'IN TRADE' : 'MONITORING') : 'STOPPED'}</span></div>
              <button onClick={() => void toggle()} disabled={busy} className={`mt-5 w-full rounded-xl py-3 font-bold ${state.running ? 'bg-rose-600 hover:bg-rose-500' : 'bg-emerald-600 hover:bg-emerald-500'}`}>{state.running ? 'STOP BOT' : 'START BOT'}</button>
              <p className="mt-3 text-xs text-slate-400">Start and Stop control paper simulation, not broker execution. Stop pauses future entries and candle advancement; any open paper position remains open.</p>
              <div className="mt-5 space-y-3 border-t border-white/10 pt-5">
                {draft && ([
                  ['Risk per trade (%)', 'risk_pct', 0.1, 1, 0.1],
                  ['Stop loss (pips)', 'stop_pips', 2, 200, 1],
                  ['Take profit (pips)', 'take_pips', 2, 400, 1],
                  ['Minimum signal confidence', 'confidence_threshold', 0.5, 0.99, 0.01],
                ] as const).map(([name,key,min,max,step]) => <label key={key} className="block text-xs text-slate-400">
                  <span className="mb-1 flex justify-between"><span>{name}</span><strong className="text-white">{draft[key]}</strong></span>
                  <input className="w-full accent-cyan-400" type="range" min={min} max={max} step={step} value={draft[key]} onChange={e => setDraft({ ...draft, [key]: Number(e.target.value) })} />
                </label>)}
                <button disabled={busy} onClick={() => void saveSettings()} className={`w-full ${button}`}>Save risk settings</button>
              </div>
              <div className="mt-5 border-t border-white/10 pt-4 text-sm">
                <p className="text-slate-400">Signal <strong className="ml-2 text-white">{state.signal.side}</strong></p>
                <p className="mt-2 text-slate-400">Heuristic strength <strong className="ml-2 text-white">{(state.signal.confidence * 100).toFixed(1)}%</strong></p>
                <p className="mt-2 text-slate-400">RSI <strong className="ml-2 text-white">{state.signal.rsi}</strong></p>
                <p className="mt-4 text-xs text-amber-300">{state.model_status}. Signal strength is not a calibrated probability of a winning trade.</p>
                {state.position && <p className="mt-3 text-xs text-cyan-300">Open {state.position.side} @ {state.position.entry.toFixed(5)} · SL {state.position.stop.toFixed(5)} · TP {state.position.target.toFixed(5)}</p>}
              </div>
            </section>
          </div>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {[
              ['Average win', money(state.metrics.avg_win)],
              ['Average loss', money(state.metrics.avg_loss)],
              ['Profit factor', state.metrics.profit_factor?.toFixed(2) ?? 'N/A'],
              ['Maximum drawdown', state.metrics.max_drawdown_pct.toFixed(2) + '%'],
            ].map(([name,value]) => <div key={name} className={panel}><p className="text-xs text-slate-400">{name}</p><p className="mt-2 text-xl font-semibold">{value}</p></div>)}
          </div>
          <section className={panel}>
            <div className="flex flex-wrap items-center justify-between gap-3"><div><h2 className="text-lg font-semibold">Paper trade history</h2><p className="mt-1 text-xs text-slate-400">{state.metrics.win_rate >= 80 && state.metrics.trades > 0 ? '80% win-rate target met in this simulation' : 'Below the 80% win-rate target / not enough evidence'}</p></div>
              <div className="flex flex-wrap items-center gap-2">
                <select aria-label="Trade filter" value={filter} onChange={e => setFilter(e.target.value)} className="rounded-lg border border-white/15 bg-[#101d34] p-2 text-sm">
                  {['ALL','WIN','LOSS','LONG','SHORT'].map(f => <option key={f} value={f}>{f}</option>)}
                </select>
                <button onClick={exportCsv} className={button}>Export CSV</button>
              </div>
            </div>
            <div className="mt-5 overflow-x-auto"><table className="w-full min-w-[1050px] text-left text-xs">
              <thead className="border-b border-white/10 text-slate-400"><tr>{['#','Opened','Closed','Side','Entry','Exit','Units','Stop','Target','P&L','Return','Result','Confidence','Reason'].map(h => <th key={h} className="px-2 py-3">{h}</th>)}</tr></thead>
              <tbody>{trades.map(t => <tr key={t.id} className="border-b border-white/5">
                <td className="px-2 py-3">{t.id}</td><td className="px-2">{stamp(t.entry_time)}</td><td className="px-2">{stamp(t.exit_time)}</td>
                <td className="px-2">{t.side}</td><td className="px-2">{t.entry.toFixed(5)}</td><td className="px-2">{t.exit.toFixed(5)}</td>
                <td className="px-2">{t.units.toFixed(2)}</td><td className="px-2">{t.stop.toFixed(5)}</td><td className="px-2">{t.target.toFixed(5)}</td>
                <td className={`px-2 ${t.pnl >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>{money(t.pnl)}</td>
                <td className="px-2">{t.pnl_pct.toFixed(2)}%</td><td className="px-2">{t.result}</td><td className="px-2">{(t.confidence * 100).toFixed(0)}%</td><td className="px-2">{t.reason}</td>
              </tr>)}</tbody></table>{!trades.length && <p className="py-8 text-center text-sm text-slate-500">No closed paper trades yet for this filter.</p>}</div>
          </section>
          <p className="pb-6 text-xs leading-6 text-slate-500">Paper trading only. Past simulated performance does not guarantee future results. Win-rate targets are objectives, not guarantees. This prototype uses synthetic prices and rule-based signals, not a fine-tuned Bitcoin model. Backend state persists in a local JSON file and should be replaced with user-scoped storage before multi-user deployment.</p>
        </>}
      </div>
    </main>
  </div></div>
}
