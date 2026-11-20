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
  effectiveEntryThreshold: number
  tradeSelectorProbability: number
  tradeSelectorSamples: number
  tradeSelectorReady: boolean
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
  heartbeatAgeSeconds: number | null
  prequentialAccuracyPct: number
  balancedAccuracyPct: number
  highConfidenceAccuracyPct: number
  evaluationSamples: number
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


export type StagedForwardResearch = {
  available: boolean
  frozenAt: string
  modelName: string
  researchOnly: boolean
  minimumNetProfitPct: number
  newSnapshots: number
  evaluableForwardRows: number
  selectedPredictions: number
  coveragePct: number
  qualifiedTradeRatePct: number | null
  avgSelectedNetReturnPct: number | null
  minimumCheckpoint: number
  preferredCheckpoint: number
  minimumCheckpointProgressPct: number
  preferredCheckpointProgressPct: number
  verdict:
    | 'NOT_FROZEN'
    | 'INSUFFICIENT_EVIDENCE'
    | 'PROMISING'
    | 'FAILING_SO_FAR'
  verdictNote: string
  evidenceStatus: string
}


export type TenMinuteForwardCandidateMetrics = {
  available: boolean
  decisions: number
  closedTrades: number
  winRatePct: number
  netReturnPct: number
  latestSignalTime: string
  latestSignal: TradingSignal | 'SKIP_FILTER'
}

export type TenMinuteV2Comparison = {
  researchOnly: boolean
  candidateName: string
  rule: string
  v1: TenMinuteForwardCandidateMetrics
  v2: TenMinuteForwardCandidateMetrics
  note: string
}


export type V3CandidateMetrics = {
  timeframe: string
  horizonMinutes: number
  proposerThreshold: number
  selectorThreshold: number
  validationTrades: number
  validationTradesPerDay: number
  validationWinRatePct: number
  validationNetMidPct: number
  validationNetHighPct: number
  testTrades: number
  testTradesPerDay: number
  testWinRatePct: number
  testNetMidPct: number
  testNetHighPct: number
  testTargetMet: boolean
}

export type V3ProfitFirstResearch = {
  available: boolean
  researchOnly: boolean
  status:
    | 'NOT_RUN'
    | 'INVALID_OUTPUT'
    | 'TARGET_MET'
    | 'TARGET_NOT_MET'
  message: string
  architecture?: string
  candidateCombinations?: number
  timeframes?: string[]
  horizonsMinutes?: number[]
  targets?: Record<string, number>
  bestCandidate: V3CandidateMetrics | null
  topCandidates: V3CandidateMetrics[]
}


export type V3TrackMetrics = {
  trades: number
  tradesPerDay: number
  winRatePct: number
  netMidPct: number
  netHighPct: number
}

export type V3ResearchTrack = {
  available: boolean
  kind:
    | 'multitimeframe'
    | 'barrier'
    | 'transformer'
  architecture?: string
  configuration?: Record<string, string | number>
  validation?: V3TrackMetrics
  test?: V3TrackMetrics & {
    targetMet: boolean
  }
}

export type V3ResearchSuite = {
  available: boolean
  researchOnly: boolean
  promotionStatus:
    | 'NOT_EVALUATED'
    | 'NO_QUALIFIED_CANDIDATE'
    | 'QUALIFIED_FOR_FORWARD_PAPER'
  requirements: Record<string, number | boolean>
  qualifiedCandidateCount: number
  selectedCandidate: Record<string, unknown> | null
  note: string
  tracks: V3ResearchTrack[]
}
