import type {
  V3ResearchSuite,
} from '../types/trading'


const API_BASE_URL =
  'http://127.0.0.1:8000'


export const getV3ResearchSuite =
  async (): Promise<V3ResearchSuite> => {
    const response =
      await fetch(
        `${API_BASE_URL}/api/research/v3-suite`,
      )

    if (!response.ok) {
      throw new Error(
        `V3 research API failed: ${response.status}`,
      )
    }

    return response.json()
  }
