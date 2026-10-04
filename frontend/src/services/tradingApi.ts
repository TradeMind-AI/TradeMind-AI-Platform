import type {
  TradingDashboardData,
} from '../types/trading'


const API_BASE_URL =
  'http://127.0.0.1:8000'


export const getDashboardData =
  async (): Promise<TradingDashboardData> => {
    const response =
      await fetch(
        `${API_BASE_URL}/api/dashboard`,
      )

    if (!response.ok) {
      throw new Error(
        `Dashboard API failed: ${response.status}`,
      )
    }

    const data:
      TradingDashboardData =
      await response.json()

    return data
  }