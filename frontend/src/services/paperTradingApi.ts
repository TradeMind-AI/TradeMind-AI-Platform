import type {
  ManualPaperAccount,
} from '../types/trading'


const API_BASE_URL =
  'http://127.0.0.1:8000'


const parseResponse =
  async (
    response: Response,
  ): Promise<ManualPaperAccount> => {
    const body =
      await response.json()
        .catch(() => null)

    if (!response.ok) {
      throw new Error(
        body?.detail
        ?? (
          'Paper trading API failed: '
          + response.status
        ),
      )
    }

    return body as ManualPaperAccount
  }


export const getManualPaperAccount =
  async (): Promise<ManualPaperAccount> => {
    const response =
      await fetch(
        API_BASE_URL
        + '/api/paper/manual/account',
      )

    return parseResponse(
      response,
    )
  }


export const placeManualPaperOrder =
  async (
    side: 'BUY' | 'SELL',
    amountUsdt: number,
  ): Promise<ManualPaperAccount> => {
    const response =
      await fetch(
        API_BASE_URL
        + '/api/paper/manual/order',
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            side,
            amount_usdt: amountUsdt,
          }),
        },
      )

    return parseResponse(
      response,
    )
  }


export const resetManualPaperAccount =
  async (): Promise<ManualPaperAccount> => {
    const response =
      await fetch(
        API_BASE_URL
        + '/api/paper/manual/reset',
        {
          method: 'POST',
        },
      )

    return parseResponse(
      response,
    )
  }
