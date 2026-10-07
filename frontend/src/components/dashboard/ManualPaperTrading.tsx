import {
  useEffect,
  useState,
} from 'react'

import {
  getManualPaperAccount,
  placeManualPaperOrder,
  resetManualPaperAccount,
} from '../../services/paperTradingApi'

import type {
  ManualPaperAccount,
} from '../../types/trading'


const money = (
  value: number,
) => value.toLocaleString(
  'en-US',
  {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  },
)


const ManualPaperTrading = () => {
  const [
    account,
    setAccount,
  ] = useState<ManualPaperAccount | null>(
    null,
  )

  const [
    amount,
    setAmount,
  ] = useState('100')

  const [
    loading,
    setLoading,
  ] = useState(true)

  const [
    submitting,
    setSubmitting,
  ] = useState(false)

  const [
    error,
    setError,
  ] = useState<string | null>(
    null,
  )


  useEffect(() => {
    let active = true

    const refresh = async () => {
      try {
        const next =
          await getManualPaperAccount()

        if (active) {
          setAccount(next)
          setError(null)
        }
      } catch (err) {
        if (active) {
          setError(
            err instanceof Error
              ? err.message
              : 'Failed to load paper account',
          )
        }
      } finally {
        if (active) {
          setLoading(false)
        }
      }
    }

    refresh()

    const timer =
      window.setInterval(
        refresh,
        5_000,
      )

    return () => {
      active = false

      window.clearInterval(
        timer,
      )
    }
  }, [])


  const submitOrder = async (
    side: 'BUY' | 'SELL',
  ) => {
    const numericAmount =
      Number(amount)

    if (
      !Number.isFinite(
        numericAmount,
      )
      || numericAmount <= 0
    ) {
      setError(
        'Enter a positive USDT amount.',
      )
      return
    }

    try {
      setSubmitting(true)
      setError(null)

      const next =
        await placeManualPaperOrder(
          side,
          numericAmount,
        )

      setAccount(next)
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : 'Paper order failed',
      )
    } finally {
      setSubmitting(false)
    }
  }


  const resetAccount = async () => {
    const confirmed = window.confirm(
      'Reset the manual paper account to $10,000 and clear all paper positions?',
    )

    if (!confirmed) {
      return
    }

    try {
      setSubmitting(true)
      setError(null)

      const next =
        await resetManualPaperAccount()

      setAccount(next)
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : 'Reset failed',
      )
    } finally {
      setSubmitting(false)
    }
  }


  return (
    <section className="rounded-2xl border border-white/10 bg-white/[0.025] p-6">
      <div className="flex flex-col gap-4 border-b border-white/5 pb-5 md:flex-row md:items-start md:justify-between">
        <div>
          <p className="text-xs uppercase tracking-[0.2em] text-white/30">
            Manual Paper Trading
          </p>

          <h2 className="mt-2 text-xl font-semibold">
            Trade BTC with simulated USDT
          </h2>

          <p className="mt-2 max-w-2xl text-sm leading-6 text-white/35">
            BUY and SELL use the live BTC/USDT market price, but no real Binance order is sent.
          </p>
        </div>

        <button
          type="button"
          onClick={resetAccount}
          disabled={submitting}
          className="rounded-lg border border-white/10 px-3 py-2 text-xs text-white/45 transition hover:bg-white/[0.05] hover:text-white disabled:opacity-40"
        >
          Reset paper account
        </button>
      </div>

      {loading && !account && (
        <div className="py-8 text-sm text-white/35">
          Loading paper account...
        </div>
      )}

      {account && (
        <>
          <div className="mt-5 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
            <div className="rounded-xl border border-white/8 bg-black/10 p-4">
              <p className="text-[10px] uppercase tracking-wider text-white/25">
                Equity
              </p>

              <p className="mt-2 text-lg font-semibold">
                {'$' + money(account.equityUsdt)}
              </p>
            </div>

            <div className="rounded-xl border border-white/8 bg-black/10 p-4">
              <p className="text-[10px] uppercase tracking-wider text-white/25">
                Cash USDT
              </p>

              <p className="mt-2 text-lg font-semibold">
                {'$' + money(account.cashUsdt)}
              </p>
            </div>

            <div className="rounded-xl border border-white/8 bg-black/10 p-4">
              <p className="text-[10px] uppercase tracking-wider text-white/25">
                BTC holdings
              </p>

              <p className="mt-2 text-lg font-semibold">
                {account.btcQuantity.toFixed(6)} BTC
              </p>

              <p className="mt-1 text-xs text-white/30">
                {'$' + money(account.btcMarketValueUsdt) + ' market value'}
              </p>
            </div>

            <div className="rounded-xl border border-white/8 bg-black/10 p-4">
              <p className="text-[10px] uppercase tracking-wider text-white/25">
                P&amp;L
              </p>

              <p className="mt-2 text-sm text-white/70">
                {'Realized $' + money(account.realizedPnlUsdt)}
              </p>

              <p className="mt-1 text-sm text-white/70">
                {'Unrealized $' + money(account.unrealizedPnlUsdt)}
              </p>
            </div>
          </div>

          <div className="mt-5 grid gap-5 lg:grid-cols-[0.8fr_1.2fr]">
            <div className="rounded-xl border border-white/8 bg-black/10 p-5">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <p className="text-xs text-white/30">
                    Live BTC price
                  </p>

                  <p className="mt-1 text-xl font-semibold">
                    {'$' + money(account.livePrice)}
                  </p>
                </div>

                <div className="text-right text-xs text-white/25">
                  {'Paper fee ' + account.feeRatePct.toFixed(3) + '% per side'}
                </div>
              </div>

              <label className="mt-5 block text-xs text-white/35">
                Order size (USDT)
              </label>

              <input
                value={amount}
                onChange={(event) =>
                  setAmount(
                    event.target.value,
                  )
                }
                type="number"
                min="1"
                step="10"
                className="mt-2 w-full rounded-lg border border-white/10 bg-black/20 px-4 py-3 text-white outline-none transition focus:border-white/25"
              />

              <div className="mt-4 grid grid-cols-2 gap-3">
                <button
                  type="button"
                  disabled={submitting}
                  onClick={() =>
                    submitOrder('BUY')
                  }
                  className="rounded-lg bg-emerald-500 px-4 py-3 text-sm font-semibold text-black transition hover:bg-emerald-400 disabled:opacity-40"
                >
                  BUY BTC
                </button>

                <button
                  type="button"
                  disabled={submitting}
                  onClick={() =>
                    submitOrder('SELL')
                  }
                  className="rounded-lg bg-red-500 px-4 py-3 text-sm font-semibold text-white transition hover:bg-red-400 disabled:opacity-40"
                >
                  SELL BTC
                </button>
              </div>

              {account.btcQuantity > 0 && (
                <p className="mt-3 text-xs text-white/30">
                  {'Average entry $' + money(account.averageEntryPrice)}
                </p>
              )}

              {error && (
                <div className="mt-4 rounded-lg border border-amber-400/20 bg-amber-400/[0.05] px-3 py-2 text-xs text-amber-200">
                  {error}
                </div>
              )}
            </div>

            <div className="rounded-xl border border-white/8 bg-black/10 p-5">
              <p className="text-xs uppercase tracking-[0.18em] text-white/25">
                Recent manual paper orders
              </p>

              <div className="mt-4 space-y-2">
                {account.recentOrders.length === 0 && (
                  <p className="py-6 text-sm text-white/25">
                    No manual paper orders yet.
                  </p>
                )}

                {account.recentOrders.slice(0, 8).map(
                  (order) => (
                    <div
                      key={
                        order.timestamp
                        + '-'
                        + order.side
                        + '-'
                        + order.price
                      }
                      className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-white/5 px-3 py-2 text-xs"
                    >
                      <div className="flex items-center gap-3">
                        <span
                          className={
                            order.side === 'BUY'
                              ? 'font-semibold text-emerald-300'
                              : 'font-semibold text-red-300'
                          }
                        >
                          {order.side}
                        </span>

                        <span className="text-white/45">
                          {'$' + money(order.amountUsdt)}
                        </span>
                      </div>

                      <div className="text-right text-white/30">
                        {'$'
                          + money(order.price)
                          + ' · '
                          + new Date(order.timestamp)
                            .toLocaleString()}
                      </div>
                    </div>
                  ),
                )}
              </div>
            </div>
          </div>
        </>
      )}

      {!account && error && (
        <div className="mt-4 rounded-lg border border-amber-400/20 bg-amber-400/[0.05] px-3 py-2 text-sm text-amber-200">
          {error}
        </div>
      )}
    </section>
  )
}


export default ManualPaperTrading
