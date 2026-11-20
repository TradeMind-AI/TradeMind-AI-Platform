import type {
  TenMinuteV2Comparison,
} from '../types/trading'


const API_BASE_URL =
  'http://127.0.0.1:8000'


export const getTenMinuteV2Comparison =
  async (): Promise<TenMinuteV2Comparison> => {
    const response =
      await fetch(
        `${API_BASE_URL}/api/research/10m-v2-comparison`,
      )

    if (!response.ok) {
      throw new Error(
        `10m V2 comparison API failed: ${response.status}`,
      )
    }

    return response.json()
  }
