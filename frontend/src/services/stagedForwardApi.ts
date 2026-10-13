import type {
  StagedForwardResearch,
} from '../types/trading'


const API_BASE_URL =
  'http://127.0.0.1:8000'


export const getStagedForwardResearch =
  async (): Promise<StagedForwardResearch> => {
    const response =
      await fetch(
        API_BASE_URL
        + '/api/research/staged-forward',
      )

    const body =
      await response.json()
        .catch(() => null)

    if (!response.ok) {
      throw new Error(
        body?.detail
        ?? (
          'Forward research API failed: '
          + response.status
        ),
      )
    }

    return body as StagedForwardResearch
  }
