import {
  useMemo,
  useState,
} from 'react'

import type {
  BTCCandle,
} from './BTCChart'


type StrategySignal =
  | 'BUY'
  | 'SELL'
  | 'HOLD'
  | 'WARMING'


type StrategyResult = {
  name: string
  signal: StrategySignal
  detail: string
}


const sma = (
  values: number[],
  period: number,
) => {
  if (values.length < period) {
    return null
  }

  const window =
    values.slice(
      -period,
    )

  return (
    window.reduce(
      (sum, value) =>
        sum + value,
      0,
    )
    / period
  )
}


const emaSeries = (
  values: number[],
  period: number,
) => {
  if (values.length < period) {
    return []
  }

  const result:
    number[] = []

  let current =
    values
      .slice(
        0,
        period,
      )
      .reduce(
        (sum, value) =>
          sum + value,
        0,
      )
    / period

  for (
    let index = 0;
    index < values.length;
    index += 1
  ) {
    if (index < period - 1) {
      result.push(
        Number.NaN,
      )
      continue
    }

    if (index === period - 1) {
      result.push(
        current,
      )
      continue
    }

    const multiplier =
      2 / (period + 1)

    current =
      (
        values[index]
        - current
      )
      * multiplier
      + current

    result.push(
      current,
    )
  }

  return result
}


const latestFinite = (
  values: number[],
) => {
  for (
    let index =
      values.length - 1;
    index >= 0;
    index -= 1
  ) {
    const value =
      values[index]

    if (
      Number.isFinite(
        value,
      )
    ) {
      return value
    }
  }

  return null
}


const rsi14 = (
  closes: number[],
) => {
  const period = 14

  if (
    closes.length
    <= period
  ) {
    return null
  }

  let gains = 0
  let losses = 0

  for (
    let index = 1;
    index <= period;
    index += 1
  ) {
    const change =
      closes[index]
      - closes[index - 1]

    if (change >= 0) {
      gains += change
    } else {
      losses += -change
    }
  }

  let averageGain =
    gains / period

  let averageLoss =
    losses / period

  for (
    let index =
      period + 1;
    index < closes.length;
    index += 1
  ) {
    const change =
      closes[index]
      - closes[index - 1]

    averageGain =
      (
        averageGain
        * (
          period - 1
        )
        + Math.max(
          change,
          0,
        )
      )
      / period

    averageLoss =
      (
        averageLoss
        * (
          period - 1
        )
        + Math.max(
          -change,
          0,
        )
      )
      / period
  }

  if (averageLoss === 0) {
    return 100
  }

  const rs =
    averageGain
    / averageLoss

  return (
    100
    - 100 / (1 + rs)
  )
}


const atr14 = (
  candles: BTCCandle[],
) => {
  if (candles.length < 15) {
    return null
  }

  const trueRanges:
    number[] = []

  for (
    let index = 1;
    index < candles.length;
    index += 1
  ) {
    const candle =
      candles[index]

    const previous =
      candles[index - 1]

    trueRanges.push(
      Math.max(
        candle.high
        - candle.low,
        Math.abs(
          candle.high
          - previous.close,
        ),
        Math.abs(
          candle.low
          - previous.close,
        ),
      ),
    )
  }

  return sma(
    trueRanges,
    14,
  )
}


const macd = (
  closes: number[],
) => {
  const fast =
    emaSeries(
      closes,
      12,
    )

  const slow =
    emaSeries(
      closes,
      26,
    )

  const macdValues =
    closes.map(
      (
        _,
        index,
      ) => {
        const fastValue =
          fast[index]

        const slowValue =
          slow[index]

        if (
          !Number.isFinite(
            fastValue,
          )
          || !Number.isFinite(
            slowValue,
          )
        ) {
          return Number.NaN
        }

        return (
          fastValue
          - slowValue
        )
      },
    )

  const clean =
    macdValues.filter(
      Number.isFinite,
    )

  const signalSeries =
    emaSeries(
      clean,
      9,
    )

  return {
    macd:
      latestFinite(
        macdValues,
      ),
    signal:
      latestFinite(
        signalSeries,
      ),
  }
}


const vwap = (
  candles: BTCCandle[],
) => {
  if (candles.length === 0) {
    return null
  }

  let totalValue = 0
  let totalVolume = 0

  for (const candle of candles) {
    const typical =
      (
        candle.high
        + candle.low
        + candle.close
      )
      / 3

    totalValue +=
      typical
      * candle.volume

    totalVolume +=
      candle.volume
  }

  if (totalVolume <= 0) {
    return null
  }

  return (
    totalValue
    / totalVolume
  )
}


const bollinger = (
  closes: number[],
) => {
  if (closes.length < 20) {
    return {
      middle: null,
      upper: null,
      lower: null,
      widthPct: null,
    }
  }

  const window =
    closes.slice(-20)

  const middle =
    window.reduce(
      (sum, value) =>
        sum + value,
      0,
    )
    / window.length

  const variance =
    window.reduce(
      (
        sum,
        value,
      ) =>
        sum
        + (
          value
          - middle
        ) ** 2,
      0,
    )
    / window.length

  const deviation =
    Math.sqrt(
      variance,
    )

  const upper =
    middle
    + 2 * deviation

  const lower =
    middle
    - 2 * deviation

  return {
    middle,
    upper,
    lower,
    widthPct:
      (
        upper
        - lower
      )
      / middle
      * 100,
  }
}


const valueText = (
  value: number | null,
  digits = 2,
) => (
  value === null
    ? '—'
    : value.toFixed(
        digits,
      )
)


const signalClass = (
  signal: StrategySignal,
) => {
  if (signal === 'BUY') {
    return (
      'border-emerald-400/20 '
      + 'bg-emerald-400/[0.06] '
      + 'text-emerald-300'
    )
  }

  if (signal === 'SELL') {
    return (
      'border-rose-400/20 '
      + 'bg-rose-400/[0.06] '
      + 'text-rose-300'
    )
  }

  if (signal === 'WARMING') {
    return (
      'border-white/10 '
      + 'bg-white/[0.03] '
      + 'text-white/35'
    )
  }

  return (
    'border-[#d4af37]/15 '
    + 'bg-[#d4af37]/[0.04] '
    + 'text-[#cdbb7b]'
  )
}


const TechnicalAnalysisPanel = ({
  candles,
}: {
  candles: BTCCandle[]
}) => {
  const [
    tab,
    setTab,
  ] = useState<
    'metrics' | 'strategies'
  >('metrics')


  const analysis =
    useMemo(
      () => {
        const closes =
          candles.map(
            (candle) =>
              candle.close,
          )

        const latest =
          candles[
            candles.length - 1
          ]

        const ema20 =
          latestFinite(
            emaSeries(
              closes,
              20,
            ),
          )

        const ema50 =
          latestFinite(
            emaSeries(
              closes,
              50,
            ),
          )

        const ema200 =
          latestFinite(
            emaSeries(
              closes,
              200,
            ),
          )

        const sma20 =
          sma(
            closes,
            20,
          )

        const rsi =
          rsi14(
            closes,
          )

        const atr =
          atr14(
            candles,
          )

        const macdValue =
          macd(
            closes,
          )

        const currentVwap =
          vwap(
            candles,
          )

        const bands =
          bollinger(
            closes,
          )

        const recentVolumes =
          candles
            .slice(
              -20,
            )
            .map(
              (candle) =>
                candle.volume,
            )

        const volumeAverage =
          sma(
            recentVolumes,
            Math.min(
              20,
              recentVolumes.length,
            ),
          )

        const relativeVolume =
          (
            latest
            && volumeAverage
            && volumeAverage > 0
          )
            ? (
                latest.volume
                / volumeAverage
              )
            : null

        const strategies:
          StrategyResult[] = []

        strategies.push({
          name:
            'EMA 20 / 50 Crossover',
          signal:
            (
              ema20 === null
              || ema50 === null
            )
              ? 'WARMING'
              : (
                  ema20 > ema50
                    ? 'BUY'
                    : 'SELL'
                ),
          detail:
            (
              ema20 === null
              || ema50 === null
            )
              ? 'Waiting for enough candles.'
              : (
                  'EMA 20 '
                  + (
                    ema20 > ema50
                      ? 'above'
                      : 'below'
                  )
                  + ' EMA 50'
                ),
        })

        strategies.push({
          name:
            'RSI Mean Reversion',
          signal:
            rsi === null
              ? 'WARMING'
              : (
                  rsi < 30
                    ? 'BUY'
                    : (
                        rsi > 70
                          ? 'SELL'
                          : 'HOLD'
                      )
                ),
          detail:
            rsi === null
              ? 'Waiting for RSI history.'
              : (
                  'RSI 14 = '
                  + rsi.toFixed(1)
                ),
        })

        strategies.push({
          name:
            'MACD Momentum',
          signal:
            (
              macdValue.macd === null
              || macdValue.signal === null
            )
              ? 'WARMING'
              : (
                  macdValue.macd
                  > macdValue.signal
                    ? 'BUY'
                    : 'SELL'
                ),
          detail:
            (
              macdValue.macd === null
              || macdValue.signal === null
            )
              ? 'Waiting for MACD history.'
              : (
                  'MACD '
                  + macdValue.macd.toFixed(2)
                  + ' vs signal '
                  + macdValue.signal.toFixed(2)
                ),
        })

        strategies.push({
          name:
            'Bollinger Breakout',
          signal:
            (
              !latest
              || bands.upper === null
              || bands.lower === null
            )
              ? 'WARMING'
              : (
                  latest.close
                  > bands.upper
                    ? 'BUY'
                    : (
                        latest.close
                        < bands.lower
                          ? 'SELL'
                          : 'HOLD'
                      )
                ),
          detail:
            bands.widthPct === null
              ? 'Waiting for Bollinger history.'
              : (
                  'Band width '
                  + bands.widthPct.toFixed(2)
                  + '%'
                ),
        })

        strategies.push({
          name:
            'VWAP Bias',
          signal:
            (
              !latest
              || currentVwap === null
            )
              ? 'WARMING'
              : (
                  latest.close
                  >= currentVwap
                    ? 'BUY'
                    : 'SELL'
                ),
          detail:
            currentVwap === null
              ? 'VWAP unavailable.'
              : (
                  'Price '
                  + (
                    latest
                    && latest.close
                    >= currentVwap
                      ? 'above'
                      : 'below'
                  )
                  + ' session VWAP'
                ),
        })

        strategies.push({
          name:
            'Long-Term Trend',
          signal:
            (
              ema50 === null
              || ema200 === null
            )
              ? 'WARMING'
              : (
                  ema50 > ema200
                    ? 'BUY'
                    : 'SELL'
                ),
          detail:
            (
              ema50 === null
              || ema200 === null
            )
              ? 'Requires at least 200 loaded candles.'
              : (
                  'EMA 50 '
                  + (
                    ema50 > ema200
                      ? 'above'
                      : 'below'
                  )
                  + ' EMA 200'
                ),
        })

        return {
          ema20,
          ema50,
          ema200,
          sma20,
          rsi,
          atr,
          macd:
            macdValue.macd,
          macdSignal:
            macdValue.signal,
          vwap:
            currentVwap,
          bollingerWidth:
            bands.widthPct,
          relativeVolume,
          strategies,
        }
      },
      [candles],
    )


  return (
    <section className="rounded-2xl border border-[#d4af37]/15 bg-[#0b0b0a] p-6">
      <div className="flex flex-col gap-4 border-b border-[#d4af37]/10 pb-5 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <p className="text-xs uppercase tracking-[0.2em] text-[#a98c45]">
            Technical Analysis
          </p>

          <h3 className="mt-2 text-xl font-semibold">
            Indicators, Metrics & Strategies
          </h3>

          <p className="mt-2 text-sm text-white/35">
            Common technical-analysis tools computed from the currently loaded chart candles.
          </p>
        </div>

        <div className="flex rounded-lg border border-[#d4af37]/15 bg-black/20 p-1">
          {([
            ['metrics', 'Metrics'],
            ['strategies', 'Strategies'],
          ] as const).map(
            ([
              key,
              label,
            ]) => (
              <button
                key={key}
                type="button"
                onClick={() =>
                  setTab(
                    key,
                  )
                }
                className={[
                  'rounded-md px-3 py-2 text-xs font-medium transition',
                  tab === key
                    ? 'bg-[linear-gradient(135deg,#f0cf68,#d4af37)] text-black'
                    : 'text-white/40 hover:text-[#f0cf68]',
                ].join(' ')}
              >
                {label}
              </button>
            ),
          )}
        </div>
      </div>

      {tab === 'metrics' && (
        <div className="mt-5 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <Metric
            label="RSI 14"
            value={valueText(
              analysis.rsi,
              1,
            )}
            note="Momentum oscillator"
          />

          <Metric
            label="MACD"
            value={valueText(
              analysis.macd,
              2,
            )}
            note={
              'Signal '
              + valueText(
                analysis.macdSignal,
                2,
              )
            }
          />

          <Metric
            label="ATR 14"
            value={valueText(
              analysis.atr,
              2,
            )}
            note="Average true range"
          />

          <Metric
            label="VWAP"
            value={
              analysis.vwap === null
                ? '—'
                : '$'
                  + analysis.vwap.toLocaleString(
                    'en-US',
                    {
                      maximumFractionDigits: 2,
                    },
                  )
            }
            note="Volume-weighted price"
          />

          <Metric
            label="EMA 20"
            value={valueText(
              analysis.ema20,
              2,
            )}
            note="Short trend"
          />

          <Metric
            label="EMA 50"
            value={valueText(
              analysis.ema50,
              2,
            )}
            note="Medium trend"
          />

          <Metric
            label="EMA 200"
            value={valueText(
              analysis.ema200,
              2,
            )}
            note="Long trend"
          />

          <Metric
            label="Bollinger width"
            value={
              analysis.bollingerWidth === null
                ? '—'
                : analysis.bollingerWidth.toFixed(2)
                  + '%'
            }
            note="20-period volatility"
          />

          <Metric
            label="Relative volume"
            value={
              analysis.relativeVolume === null
                ? '—'
                : analysis.relativeVolume.toFixed(2)
                  + '×'
            }
            note="Latest vs recent mean"
          />

          <Metric
            label="SMA 20"
            value={valueText(
              analysis.sma20,
              2,
            )}
            note="Simple moving average"
          />
        </div>
      )}

      {tab === 'strategies' && (
        <div className="mt-5 grid gap-3 md:grid-cols-2 xl:grid-cols-3">
          {analysis.strategies.map(
            (strategy) => (
              <div
                key={strategy.name}
                className="rounded-xl border border-[#d4af37]/10 bg-black/15 p-4"
              >
                <div className="flex items-start justify-between gap-4">
                  <div>
                    <p className="font-medium text-white/80">
                      {strategy.name}
                    </p>

                    <p className="mt-2 text-xs leading-5 text-white/30">
                      {strategy.detail}
                    </p>
                  </div>

                  <span
                    className={[
                      'rounded-md border px-2 py-1 text-[10px] font-semibold tracking-wider',
                      signalClass(
                        strategy.signal,
                      ),
                    ].join(' ')}
                  >
                    {strategy.signal}
                  </span>
                </div>
              </div>
            ),
          )}
        </div>
      )}

      <p className="mt-5 text-xs leading-5 text-white/25">
        Strategy cards are technical-rule signals for research and visualization only. They are not connected to real-money execution and do not imply profitability.
      </p>
    </section>
  )
}


const Metric = ({
  label,
  value,
  note,
}: {
  label: string
  value: string
  note: string
}) => (
  <div className="rounded-xl border border-[#d4af37]/10 bg-black/15 p-4">
    <p className="text-[10px] uppercase tracking-[0.16em] text-[#91783c]">
      {label}
    </p>

    <p className="mt-2 text-lg font-semibold text-[#efe3bf]">
      {value}
    </p>

    <p className="mt-1 text-xs text-white/25">
      {note}
    </p>
  </div>
)


export default TechnicalAnalysisPanel
