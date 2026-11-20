import type {
  V3ProfitFirstResearch,
} from '../types/trading'


const API_BASE_URL =
  'http://127.0.0.1:8000'


export const getV3ProfitFirstResearch =
  async (): Promise<V3ProfitFirstResearch> => {
    const response =
      await fetch(
        `${API_BASE_URL}/api/research/v3-profit-first`,
      )

    if (!response.ok) {
      throw new Error(
        `V3 research API failed: ${response.status}`,
      )
    }

    return response.json()
  }
