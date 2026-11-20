export type ForexCandle = { time: number; open: number; high: number; low: number; close: number }
export type ForexTrade = {
  id: number; side: 'LONG' | 'SHORT'; entry_time: number; exit_time: number
  entry: number; exit: number; units: number; stop: number; target: number
  confidence: number; reason: string; pnl: number; pnl_pct: number
  result: 'WIN' | 'LOSS'; reason_exit: string
}
export type ForexState = {
  running: boolean; cursor: number; balance: number
  source: string; model_status: string; paper_only: boolean
  position: null | {
    side: 'LONG' | 'SHORT'; entry: number; units: number
    stop: number; target: number; confidence: number; entry_time: number
  }
  settings: { risk_pct: number; stop_pips: number; take_pips: number; confidence_threshold: number }
  signal: { side: 'LONG' | 'SHORT' | 'HOLD'; confidence: number; rsi: number; ema20: number; ema50: number; reason: string }
  metrics: { trades: number; wins: number; win_rate: number; net_pnl: number; avg_win: number
    avg_loss: number; profit_factor: number | null; max_drawdown_pct: number; target_win_rate: number }
  trades: ForexTrade[]
}
const BASE = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'
async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}/api/forex${path}`, init)
  if (!response.ok) throw new Error(`Forex API: ${response.status} ${await response.text()}`)
  return response.json() as Promise<T>
}
export const getForexStatus = () => request<ForexState>('/status')
export const getForexCandles = () => request<{candles: ForexCandle[]; source: string}>('/candles?limit=200')
export const forexStart = () => request<ForexState>('/start', { method: 'POST' })
export const forexStop = () => request<ForexState>('/stop', { method: 'POST' })
export const forexStep = () => request<ForexState>('/step', { method: 'POST' })
export const updateForexSettings = (settings: ForexState['settings']) =>
  request<ForexState>('/settings', { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(settings) })
