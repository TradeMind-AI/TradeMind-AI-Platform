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

import {
  getDashboardData,
} from '../../services/tradingApi'

import {
  getBTCCandles,
} from '../../services/marketApi'

import type {
  TradingDashboardData,
} from '../../types/trading'


const Dashboard = () => {
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
    loading,
    setLoading,
  ] =
    useState(true)


  const [
    error,
    setError,
  ] =
    useState<string | null>(
      null,
    )


  useEffect(() => {
    const loadDashboard =
      async () => {
        try {
          setError(null)

          const [
            dashboard,
            market,
          ] =
            await Promise.all([
              getDashboardData(),
              getBTCCandles(200),
            ])

          setDashboardData(
            dashboard,
          )

          setCandles(
            market,
          )
        } catch (err) {
          setError(
            err instanceof Error
              ? err.message
              : 'Failed to load dashboard',
          )
        } finally {
          setLoading(false)
        }
      }


    loadDashboard()
  }, [])


  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-[#030712] text-white">
        <p className="text-sm text-white/40">
          Loading TradeMind dashboard...
        </p>
      </div>
    )
  }


  if (
    error
    || !dashboardData
  ) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-[#030712] text-white">
        <div className="text-center">
          <p className="text-lg font-medium">
            Dashboard unavailable
          </p>

          <p className="mt-2 text-sm text-white/40">
            {error}
          </p>
        </div>
      </div>
    )
  }


  return (
    <div className="min-h-screen bg-[#030712] text-white">
      <div className="flex">
        <Sidebar />

        <div className="min-w-0 flex-1">
          <DashboardHeader
            symbol={
              dashboardData.symbol
            }
            timeframe={
              dashboardData.timeframe
            }
            latestCandle={
              dashboardData.latestCandle
            }
          />

          <main className="px-6 py-8 lg:px-8">
            <div className="mx-auto max-w-7xl">
              <div>
                <p className="text-xs uppercase tracking-[0.2em] text-white/30">
                  Overview
                </p>

                <h2 className="mt-2 text-3xl font-semibold tracking-tight">
                  Paper Trading Dashboard
                </h2>

                <p className="mt-3 max-w-2xl text-sm leading-6 text-white/35">
                  Monitor live BTC
                  market data,
                  AI signals,
                  model confidence,
                  paper account
                  performance,
                  and system status.
                </p>
              </div>


              <div className="mt-8">
                <BTCChart
                  candles={
                    candles
                  }
                />
              </div>


              <div className="mt-6 grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
                <StatCard
                  label="BTC Price"
                  value={`$${dashboardData.price.toLocaleString(
                    'en-US',
                    {
                      minimumFractionDigits:
                        2,
                      maximumFractionDigits:
                        2,
                    },
                  )}`}
                  subtext="Latest closed 1H candle"
                />

                <StatCard
                  label="Paper Balance"
                  value={`$${dashboardData.balance.toLocaleString(
                    'en-US',
                    {
                      minimumFractionDigits:
                        2,
                      maximumFractionDigits:
                        2,
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
                <RecentSignals
                  signals={
                    dashboardData.recentSignals
                  }
                />
              </div>
            </div>
          </main>
        </div>
      </div>
    </div>
  )
}


export default Dashboard