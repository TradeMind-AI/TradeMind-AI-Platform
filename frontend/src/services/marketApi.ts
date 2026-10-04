import type {
  BTCCandle,
} from '../components/dashboard/BTCChart'


const API_BASE_URL =
  'http://127.0.0.1:8000'


export const getBTCCandles =
  async (
    limit = 200,
  ): Promise<BTCCandle[]> => {
    const response =
      await fetch(
        `${API_BASE_URL}/api/market/btc?limit=${limit}`,
      )

    if (!response.ok) {
      throw new Error(
        `BTC market API failed: ${response.status}`,
      )
    }

    const data: BTCCandle[] =
      await response.json()

    return data
  }