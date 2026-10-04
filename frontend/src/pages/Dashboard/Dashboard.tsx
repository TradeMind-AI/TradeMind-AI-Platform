import {
  useEffect,
  useState,
} from 'react'

import DashboardHeader
  from '../../components/layout/DashboardHeader'

import Sidebar
  from '../../components/layout/Sidebar'

import BTCChart, {
  type BTCCandle,
} from '../../components/dashboard/BTCChart'

import RecentSignals
  from '../../components/dashboard/RecentSignals'

import SignalCard
  from '../../components/dashboard/SignalCard'

import StatCard
  from '../../components/dashboard/StatCard'

import SystemStatus
  from '../../components/dashboard/SystemStatus'

import StrategyResearch
  from '../../components/dashboard/StrategyResearch'

import {
  getDashboardData,
  getTimeframeStatus,
} from '../../services/tradingApi'

import {
  getBTCCandles,
} from '../../services/marketApi'

import type {
  ChartTimeframe,
  StrategyTimeframe,
  TimeframeStatus,
  TradingDashboardData,
} from '../../types/trading'


const STRATEGY_TIMEFRAMES:
  StrategyTimeframe[] = [
    '1m',
    '2m',
    '5m',
    '10m',
    '20m',
    '30m',
    '1h',
  ]


const Dashboard = () => {
  const [
    selectedTimeframe,
    setSelectedTimeframe,
  ] =
    useState<StrategyTimeframe>(
      '10m',
    )

  const [
    chartTimeframe,
    setChartTimeframe,
  ] =
    useState<ChartTimeframe>(
      '1m',
    )

  const [
    timeframeStatus,
    setTimeframeStatus,
  ] =
    useState<
      TimeframeStatus[]
    >([])

  const [
    dashboardData,
    setDashboardData,
  ] =
    useState<
      TradingDashboardData | null
    >(null)

  const [
    candles,
    setCandles,
  ] =
    useState<BTCCandle[]>([])

  const [
    strategyLoading,
    setStrategyLoading,
  ] =
    useState(true)

  const [
    strategyError,
    setStrategyError,
  ] =
    useState<string | null>(
      null,
    )

  const [
    chartLoading,
    setChartLoading,
  ] =
    useState(true)

  const [
    chartError,
    setChartError,
  ] =
    useState<string | null>(
      null,
    )


  useEffect(() => {
    getTimeframeStatus()
      .then(
        setTimeframeStatus,
      )
      .catch(() => {
        setTimeframeStatus([])
      })
  }, [])


  useEffect(() => {
    let active = true

    const loadStrategy =
      async () => {
        try {
          setStrategyLoading(true)
          setStrategyError(null)

          const dashboard =
            await getDashboardData(
              selectedTimeframe,
            )

          if (!active) {
            return
          }

          setDashboardData(
            dashboard,
          )
        } catch (err) {
          if (!active) {
            return
          }

          setDashboardData(null)

          setStrategyError(
            err instanceof Error
              ? err.message
              : 'Failed to load strategy',
          )
        } finally {
          if (active) {
            setStrategyLoading(
              false,
            )
          }
        }
      }

    loadStrategy()

    const refresh =
      window.setInterval(
        loadStrategy,
        60_000,
      )

    return () => {
      active = false

      window.clearInterval(
        refresh,
      )
    }
  }, [selectedTimeframe])


  useEffect(() => {
    let active = true

    const loadChart =
      async () => {
        try {
          setChartError(null)

          if (
            candles.length === 0
          ) {
            setChartLoading(true)
          }

          const market =
            await getBTCCandles(
              chartTimeframe,
              200,
            )

          if (!active) {
            return
          }

          setCandles(
            market,
          )
        } catch (err) {
          if (!active) {
            return
          }

          setChartError(
            err instanceof Error
              ? err.message
              : 'Failed to load live chart',
          )
        } finally {
          if (active) {
            setChartLoading(
              false,
            )
          }
        }
      }

    setCandles([])
    setChartLoading(true)

    loadChart()

    const refresh =
      window.setInterval(
        loadChart,
        5_000,
      )

    return () => {
      active = false

      window.clearInterval(
        refresh,
      )
    }
  }, [chartTimeframe])


  const selectedStatus =
    timeframeStatus.find(
      (item) =>
        item.timeframe
        === selectedTimeframe,
    )


  return (
    <div className="min-h-screen bg-[#030712] text-white">
      <div className="flex">
        <Sidebar />

        <div className="min-w-0 flex-1">
          <DashboardHeader
            symbol={
              dashboardData?.symbol
              ?? 'BTC / USDT'
            }
            timeframe={
              dashboardData?.timeframe
              ?? selectedTimeframe
            }
            latestCandle={
              dashboardData?.latestCandle
              ?? 'Waiting for strategy data'
            }
          />

          <main className="px-6 py-8 lg:px-8">
            <div className="mx-auto max-w-7xl">
              <div>
                <p className="text-xs uppercase tracking-[0.2em] text-white/30">
                  Overview
                </p>

                <h2 className="mt-2 text-3xl font-semibold tracking-tight">
                  Live Market + AI Strategy
                </h2>

                <p className="mt-3 max-w-3xl text-sm leading-6 text-white/35">
                  Watch BTC on any chart interval while the
                  AI strategy keeps its own independent
                  decision timeframe.
                </p>
              </div>


              <div className="mt-6">
                <p className="mb-2 text-[10px] uppercase tracking-[0.18em] text-white/25">
                  AI strategy timeframe
                </p>

                <div className="flex flex-wrap gap-2">
                  {STRATEGY_TIMEFRAMES.map(
                    (timeframe) => {
                      const status =
                        timeframeStatus.find(
                          (item) =>
                            item.timeframe
                            === timeframe,
                        )

                      const ready =
                        status?.tradingReady
                        ?? (
                          timeframe
                          === '1h'
                        )

                      const active =
                        selectedTimeframe
                        === timeframe

                      return (
                        <button
                          key={timeframe}
                          onClick={() =>
                            setSelectedTimeframe(
                              timeframe,
                            )
                          }
                          className={[
                            'rounded-lg border px-4 py-2 text-sm transition',
                            active
                              ? 'border-white/30 bg-white text-black'
                              : 'border-white/10 bg-white/[0.03] text-white/60 hover:bg-white/[0.07] hover:text-white',
                          ].join(' ')}
                        >
                          {timeframe}

                          {!ready && (
                            <span className="ml-2 text-[10px] opacity-50">
                              setup
                            </span>
                          )}
                        </button>
                      )
                    },
                  )}
                </div>

                <p className="mt-2 text-xs text-white/25">
                  Changing the chart interval below does not
                  change the selected AI model.
                </p>
              </div>


              <div className="mt-8">
                <BTCChart
                  candles={candles}
                  timeframe={
                    chartTimeframe
                  }
                  onTimeframeChange={
                    setChartTimeframe
                  }
                  loading={
                    chartLoading
                  }
                  error={
                    chartError
                  }
                />
              </div>


              {strategyLoading && (
                <div className="mt-6 rounded-2xl border border-white/10 bg-white/[0.025] p-8 text-sm text-white/40">
                  Loading {selectedTimeframe} AI strategy...
                </div>
              )}


              {!strategyLoading
                && strategyError && (
                <div className="mt-6 rounded-2xl border border-amber-400/20 bg-amber-400/[0.05] p-6">
                  <p className="font-medium text-amber-200">
                    {selectedTimeframe} strategy is not ready yet
                  </p>

                  <p className="mt-2 text-sm leading-6 text-white/45">
                    {strategyError}
                  </p>

                  {selectedStatus && (
                    <div className="mt-4 grid gap-2 text-xs text-white/40 sm:grid-cols-4">
                      <span>
                        Market:{' '}
                        {selectedStatus.marketReady
                          ? 'ready'
                          : 'pending'}
                      </span>

                      <span>
                        Features:{' '}
                        {selectedStatus.featuresReady
                          ? 'ready'
                          : 'pending'}
                      </span>

                      <span>
                        Model:{' '}
                        {selectedStatus.modelReady
                          ? 'ready'
                          : 'pending'}
                      </span>

                      <span>
                        Paper log:{' '}
                        {selectedStatus.paperLogReady
                          ? 'ready'
                          : 'pending'}
                      </span>
                    </div>
                  )}
                </div>
              )}


              {!strategyLoading
                && dashboardData
                && !strategyError && (
                <>
                  <div className="mt-6 grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
                    <StatCard
                      label="BTC Price"
                      value={`$${dashboardData.price.toLocaleString(
                        'en-US',
                        {
                          minimumFractionDigits: 2,
                          maximumFractionDigits: 2,
                        },
                      )}`}
                      subtext={`Latest closed ${dashboardData.timeframe} strategy candle`}
                    />

                    {dashboardData.strategyMode
                      === 'Forward research'
                      ? (
                        <>
                          <StatCard
                            label="Forward Decisions"
                            value={String(
                              dashboardData.forwardSignalCount,
                            )}
                            subtext="Frozen model decisions recorded"
                          />

                          <StatCard
                            label="Closed Trades"
                            value={String(
                              dashboardData.closedForwardTrades,
                            )}
                            subtext="Completed 60m paper outcomes"
                          />

                          <StatCard
                            label="Mid-Cost Net"
                            value={`${dashboardData.netReturn.toFixed(
                              2,
                            )}%`}
                            subtext="Using 0.14% round-trip cost assumption"
                          />
                        </>
                      )
                      : (
                        <>
                          <StatCard
                            label="Paper Balance"
                            value={`$${dashboardData.balance.toLocaleString(
                              'en-US',
                              {
                                minimumFractionDigits: 2,
                                maximumFractionDigits: 2,
                              },
                            )}`}
                            subtext="Simulated account"
                          />

                          <StatCard
                            label="Win Rate"
                            value={`${dashboardData.winRate.toFixed(
                              2,
                            )}%`}
                            subtext="Closed paper trades"
                          />

                          <StatCard
                            label="Net Return"
                            value={`${dashboardData.netReturn.toFixed(
                              2,
                            )}%`}
                            subtext="After trading costs"
                          />
                        </>
                      )}
                  </div>


                  <div className="mt-6 grid gap-6 xl:grid-cols-[1.3fr_0.7fr]">
                    <SignalCard
                      prediction={
                        dashboardData.prediction
                      }
                      confidence={
                        dashboardData.confidence
                      }
                      signal={
                        dashboardData.signal
                      }
                      probabilities={
                        dashboardData.probabilities
                      }
                    />

                    <SystemStatus
                      modelName={
                        dashboardData.modelName
                      }
                      featureCount={
                        dashboardData.featureCount
                      }
                      confidenceThreshold={
                        dashboardData.confidenceThreshold
                      }
                      systemStatus={
                        dashboardData.systemStatus
                      }
                    />
                  </div>


                  <div className="mt-6">
                    <StrategyResearch
                      data={dashboardData}
                    />
                  </div>


                  <div className="mt-6">
                    <RecentSignals
                      signals={
                        dashboardData.recentSignals
                      }
                    />
                  </div>
                </>
              )}
            </div>
          </main>
        </div>
      </div>
    </div>
  )
}


export default Dashboard
