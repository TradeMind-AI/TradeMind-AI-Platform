export type TradingSignal =
  | 'BUY'
  | 'SELL'
  | 'HOLD'
  | 'SKIP_OVERLAP'

export type Prediction =
  'UP' | 'DOWN' | 'NEUTRAL'

export type StrategyTimeframe =
  | '1m'
  | '2m'
  | '5m'
  | '10m'
  | '20m'
  | '30m'
  | '1h'

export type ProbabilitySet = {
  down: number
  neutral: number
  up: number
}

export type RecentSignal = {
  timestamp: string
  prediction: Prediction
  confidence: number
  signal: TradingSignal
  status: string
}

export type TradingDashboardData = {
  symbol: string
  timeframe: string
  timeframeKey: StrategyTimeframe
  price: number
  prediction: Prediction
  confidence: number
  signal: TradingSignal
  probabilities: ProbabilitySet
  balance: number
  winRate: number
  netReturn: number
  modelName: string
  featureCount: number
  confidenceThreshold: number
  latestCandle: string
  systemStatus: string
  recentSignals: RecentSignal[]
  strategyMode: string
  researchStatus: string
  forecastHorizonMinutes: number
  targetThresholdPct: number
  entryModel: string
  exitModel: string
  primaryCostPct: number
  stressCostPct: number
  forwardSignalCount: number
  closedForwardTrades: number
  evidenceNote: string
}

export type TimeframeStatus = {
  timeframe: StrategyTimeframe
  label: string
  marketReady: boolean
  featuresReady: boolean
  modelReady: boolean
  paperLogReady: boolean
  tradingReady: boolean
}
