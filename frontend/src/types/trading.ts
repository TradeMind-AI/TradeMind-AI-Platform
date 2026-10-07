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

export type ChartTimeframe =
  | '1m'
  | '2m'
  | '3m'
  | '5m'
  | '10m'
  | '15m'
  | '20m'
  | '30m'
  | '1h'
  | '4h'
  | '1d'

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


export type AIIndicatorPoint = {
  time: number
  score: number
  up: number
  neutral: number
  down: number
  confidence: number
  prediction: Prediction
  signal: TradingSignal
}


export type ManualPaperOrder = {
  timestamp: string
  side: 'BUY' | 'SELL'
  price: number
  amountUsdt: number
  btcQuantity: number
  feeUsdt: number
  realizedPnlUsdt: number
}

export type ManualPaperAccount = {
  cashUsdt: number
  btcQuantity: number
  averageEntryPrice: number
  btcMarketValueUsdt: number
  equityUsdt: number
  realizedPnlUsdt: number
  unrealizedPnlUsdt: number
  livePrice: number
  feeRatePct: number
  recentOrders: ManualPaperOrder[]
}


export type MicroPaperTrade = {
  timestamp: string
  event: string
  side: string
  entryPrice: number
  exitPrice: number
  notionalUsdt: number
  netReturnPct: number
  pnlUsdt: number
  reason: string
  modelConfidence: number
}

export type MicroPaperLatest = {
  timestamp: string
  price: number
  prediction: number
  confidence: number
  probDown: number
  probNeutral: number
  probUp: number
  learningSamples: number
  decisionSource: string
  tradeDirection: number
  spotDepthImbalance: number
  spotTradeFlowImbalance: number
  spotVelocityBps: number
  futuresDepthImbalance: number
  futuresTradeFlowImbalance: number
  futuresVelocityBps: number
  fundingRate: number
  openInterestChangePct: number
}

export type MicroPaperPosition = {
  direction: number
  entry_price: number
  notional_usdt: number
  opened_at: string
  model_confidence: number
  learning_samples: number
  decision_source: string
}

export type MicroPaperStatus = {
  running: boolean
  message: string
  equityUsdt: number
  realizedPnlUsdt: number
  closedTrades: number
  wins: number
  losses: number
  winRatePct: number
  position: MicroPaperPosition | null
  latest: MicroPaperLatest | null
  recentTrades: MicroPaperTrade[]
}
