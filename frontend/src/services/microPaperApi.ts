import type {
  MicroPaperStatus,
} from '../types/trading'


const API_BASE_URL =
  'http://127.0.0.1:8000'


export const getMicroPaperStatus =
  async (): Promise<MicroPaperStatus> => {
    const response =
      await fetch(
        API_BASE_URL
        + '/api/paper/micro/status',
      )

    const body =
      await response.json()
        .catch(() => null)

    if (!response.ok) {
      throw new Error(
        body?.detail
        ?? (
          'Micro paper status failed: '
          + response.status
        ),
      )
    }

    return body as MicroPaperStatus
  }
