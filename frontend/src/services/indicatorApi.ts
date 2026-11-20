import type {
  AIIndicatorPoint,
} from '../types/trading'


const API_BASE_URL =
  'http://127.0.0.1:8000'


export const getAIIndicator =
  async (
    limit = 200,
  ): Promise<AIIndicatorPoint[]> => {
    const params =
      new URLSearchParams({
        timeframe: '10m',
        limit: String(limit),
      })

    const response =
      await fetch(
        `${API_BASE_URL}/api/indicator/ai?${params.toString()}`,
      )

    if (!response.ok) {
      const body =
        await response.json()
          .catch(() => null)

      throw new Error(
        body?.detail
        ?? `AI indicator API failed: ${response.status}`,
      )
    }

    return response.json()
  }
