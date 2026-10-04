import type {
  BTCCandle,
} from '../components/dashboard/BTCChart'

import type {
  StrategyTimeframe,
} from '../types/trading'


const API_BASE_URL =
  'http://127.0.0.1:8000'


export const getBTCCandles =
  async (
    timeframe:
      StrategyTimeframe = '1h',
    limit = 200,
  ): Promise<BTCCandle[]> => {
    const params =
      new URLSearchParams({
        timeframe,
        limit:
          String(limit),
      })

    const response =
      await fetch(
        `${API_BASE_URL}/api/market/btc?${params.toString()}`,
      )

    if (!response.ok) {
      const body =
        await response.json()
          .catch(() => null)

      throw new Error(
        body?.detail
        ?? `BTC market API failed: ${response.status}`,
      )
    }

    const data:
      BTCCandle[] =
      await response.json()

    return data
  }
