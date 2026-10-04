export type TradingSignal = 'BUY' | 'SELL' | 'HOLD'

export type Prediction = 'UP' | 'DOWN' | 'NEUTRAL'

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
}