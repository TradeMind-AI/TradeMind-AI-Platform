import type {
  StrategyTimeframe,
  TimeframeStatus,
  TradingDashboardData,
} from '../types/trading'


const API_BASE_URL =
  'http://127.0.0.1:8000'


export const getDashboardData =
  async (
    timeframe:
      StrategyTimeframe = '1h',
  ): Promise<TradingDashboardData> => {
    const params =
      new URLSearchParams({
        timeframe,
      })

    const response =
      await fetch(
        `${API_BASE_URL}/api/dashboard?${params.toString()}`,
      )

    if (!response.ok) {
      const body =
        await response.json()
          .catch(() => null)

      throw new Error(
        body?.detail
        ?? `Dashboard API failed: ${response.status}`,
      )
    }

    const data:
      TradingDashboardData =
      await response.json()

    return data
  }


export const getTimeframeStatus =
  async (): Promise<TimeframeStatus[]> => {
    const response =
      await fetch(
        `${API_BASE_URL}/api/timeframes`,
      )

    if (!response.ok) {
      throw new Error(
        `Timeframe API failed: ${response.status}`,
      )
    }

    return response.json()
  }
