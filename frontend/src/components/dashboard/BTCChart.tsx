import {
  CandlestickSeries,
  HistogramSeries,
  LineSeries,
  createChart,
  type CandlestickData,
  type HistogramData,
  type LineData,
  type UTCTimestamp,
} from 'lightweight-charts'

import {
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react'

import type {
  AIIndicatorPoint,
  ChartTimeframe,
} from '../../types/trading'


export type BTCCandle = {
  time: number
  open: number
  high: number
  low: number
  close: number
  volume: number
}


type CandleSeriesHandle = {
  setData: (
    data: CandlestickData[],
  ) => void
}


type VolumeSeriesHandle = {
  setData: (
    data: HistogramData[],
  ) => void
}

type LineSeriesHandle = {
  setData: (
    data: LineData[],
  ) => void
  applyOptions: (
    options: {
      visible?: boolean
    },
  ) => void
}


type BTCChartProps = {
  candles: BTCCandle[]
  timeframe: ChartTimeframe
  onTimeframeChange: (
    timeframe: ChartTimeframe,
  ) => void
  loading?: boolean
  error?: string | null
  aiPoints: AIIndicatorPoint[]
}


const CHART_TIMEFRAMES:
  ChartTimeframe[] = [
    '1m',
    '2m',
    '3m',
    '5m',
    '10m',
    '15m',
    '20m',
    '30m',
    '1h',
    '4h',
    '1d',
  ]


type IndicatorKey =
  | 'ema20'
  | 'sma20'
  | 'bollinger'
  | 'volume'
  | 'ai'


const simpleMovingAverage = (
  values: number[],
  period: number,
) => {
  const output:
    Array<number | null> =
    new Array(
      values.length,
    ).fill(null)

  let total = 0

  for (
    let index = 0;
    index < values.length;
    index += 1
  ) {
    total += values[index]

    if (index >= period) {
      total -= values[
        index - period
      ]
    }

    if (index >= period - 1) {
      output[index] =
        total / period
    }
  }

  return output
}


const exponentialMovingAverage = (
  values: number[],
  period: number,
) => {
  const output:
    Array<number | null> =
    new Array(
      values.length,
    ).fill(null)

  if (values.length < period) {
    return output
  }

  let previous =
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

  output[
    period - 1
  ] = previous

  const multiplier =
    2 / (period + 1)

  for (
    let index = period;
    index < values.length;
    index += 1
  ) {
    previous =
      (
        values[index]
        - previous
      )
      * multiplier
      + previous

    output[index] =
      previous
  }

  return output
}


const indicatorLineData = (
  candles: BTCCandle[],
  values:
    Array<number | null>,
): LineData[] =>
  candles.flatMap(
    (
      candle,
      index,
    ) => {
      const value =
        values[index]

      if (
        value === null
        || !Number.isFinite(
          value,
        )
      ) {
        return []
      }

      return [
        {
          time:
            candle.time as UTCTimestamp,
          value,
        },
      ]
    },
  )


const BTCChart = ({
  candles,
  timeframe,
  onTimeframeChange,
  loading = false,
  error = null,
  aiPoints,
}: BTCChartProps) => {
  const [
    indicators,
    setIndicators,
  ] = useState<
    Record<
      IndicatorKey,
      boolean
    >
  >({
    ema20: true,
    sma20: false,
    bollinger: false,
    volume: true,
    ai: true,
  })

  const [
    indicatorMenuOpen,
    setIndicatorMenuOpen,
  ] = useState(false)

  const chartContainerRef =
    useRef<HTMLDivElement | null>(null)

  const chartRef =
    useRef<ReturnType<
      typeof createChart
    > | null>(null)

  const candleSeriesRef =
    useRef<CandleSeriesHandle | null>(
      null,
    )

  const volumeSeriesRef =
    useRef<VolumeSeriesHandle | null>(
      null,
    )

  const aiSeriesRef =
    useRef<LineSeriesHandle | null>(
      null,
    )

  const emaSeriesRef =
    useRef<LineSeriesHandle | null>(
      null,
    )

  const smaSeriesRef =
    useRef<LineSeriesHandle | null>(
      null,
    )

  const bollingerUpperRef =
    useRef<LineSeriesHandle | null>(
      null,
    )

  const bollingerLowerRef =
    useRef<LineSeriesHandle | null>(
      null,
    )

  const lastTimeframeRef =
    useRef<ChartTimeframe | null>(
      null,
    )


  const computedIndicators =
    useMemo(
      () => {
        const closes =
          candles.map(
            (candle) =>
              candle.close,
          )

        const sma20 =
          simpleMovingAverage(
            closes,
            20,
          )

        const ema20 =
          exponentialMovingAverage(
            closes,
            20,
          )

        const upper:
          Array<number | null> =
          new Array(
            closes.length,
          ).fill(null)

        const lower:
          Array<number | null> =
          new Array(
            closes.length,
          ).fill(null)

        for (
          let index = 19;
          index < closes.length;
          index += 1
        ) {
          const mean =
            sma20[index]

          if (mean === null) {
            continue
          }

          const window =
            closes.slice(
              index - 19,
              index + 1,
            )

          const variance =
            window.reduce(
              (
                total,
                value,
              ) =>
                total
                + (
                  value
                  - mean
                ) ** 2,
              0,
            )
            / window.length

          const deviation =
            Math.sqrt(
              variance,
            )

          upper[index] =
            mean
            + 2 * deviation

          lower[index] =
            mean
            - 2 * deviation
        }

        return {
          ema20:
            indicatorLineData(
              candles,
              ema20,
            ),
          sma20:
            indicatorLineData(
              candles,
              sma20,
            ),
          upper:
            indicatorLineData(
              candles,
              upper,
            ),
          lower:
            indicatorLineData(
              candles,
              lower,
            ),
        }
      },
      [candles],
    )


  useEffect(() => {
    if (!chartContainerRef.current) {
      return
    }

    const container =
      chartContainerRef.current

    const chart = createChart(
      container,
      {
        width: container.clientWidth,
        height: 520,

        layout: {
          background: {
            color: '#080807',
          },
          textColor:
            'rgba(232,217,173,0.65)',
        },

        grid: {
          vertLines: {
            color:
              'rgba(212,175,55,0.045)',
          },
          horzLines: {
            color:
              'rgba(212,175,55,0.045)',
          },
        },

        crosshair: {
          vertLine: {
            color:
              'rgba(240,207,104,0.28)',
            labelBackgroundColor:
              '#17130a',
          },
          horzLine: {
            color:
              'rgba(240,207,104,0.28)',
            labelBackgroundColor:
              '#17130a',
          },
        },

        rightPriceScale: {
          borderColor:
            'rgba(212,175,55,0.12)',
        },

        timeScale: {
          borderColor:
            'rgba(212,175,55,0.12)',
          timeVisible: true,
          secondsVisible: false,
          rightOffset: 4,
          barSpacing: 8,
        },
      },
    )

    chartRef.current = chart

    candleSeriesRef.current =
      chart.addSeries(
        CandlestickSeries,
        {
          upColor: '#22c55e',
          downColor: '#ef4444',
          borderVisible: false,
          wickUpColor: '#22c55e',
          wickDownColor: '#ef4444',
          priceLineVisible: true,
          lastValueVisible: true,
        },
      )

    const volumeSeries =
      chart.addSeries(
        HistogramSeries,
        {
          priceScaleId: '',
          priceFormat: {
            type: 'volume',
          },
        },
      )

    volumeSeries
      .priceScale()
      .applyOptions({
        scaleMargins: {
          top: 0.8,
          bottom: 0,
        },
      })

    volumeSeriesRef.current =
      volumeSeries

    const aiSeries =
      chart.addSeries(
        LineSeries,
        {
          priceScaleId: 'ai',
          color: '#c89b3c',
          lineWidth: 2,
          priceLineVisible: false,
          lastValueVisible: false,
        },
      )

    chart.priceScale('ai')
      .applyOptions({
        visible: false,
        scaleMargins: {
          top: 0.08,
          bottom: 0.58,
        },
      })

    aiSeriesRef.current =
      aiSeries

    emaSeriesRef.current =
      chart.addSeries(
        LineSeries,
        {
          color: '#d89c45',
          lineWidth: 2,
          priceLineVisible: false,
          lastValueVisible: false,
        },
      )

    smaSeriesRef.current =
      chart.addSeries(
        LineSeries,
        {
          color: '#e7c85e',
          lineWidth: 2,
          priceLineVisible: false,
          lastValueVisible: false,
        },
      )

    bollingerUpperRef.current =
      chart.addSeries(
        LineSeries,
        {
          color:
            'rgba(213,190,125,0.62)',
          lineWidth: 1,
          priceLineVisible: false,
          lastValueVisible: false,
        },
      )

    bollingerLowerRef.current =
      chart.addSeries(
        LineSeries,
        {
          color:
            'rgba(213,190,125,0.62)',
          lineWidth: 1,
          priceLineVisible: false,
          lastValueVisible: false,
        },
      )

    const resizeObserver =
      new ResizeObserver(() => {
        chart.applyOptions({
          width:
            container.clientWidth,
        })
      })

    resizeObserver.observe(
      container,
    )

    return () => {
      resizeObserver.disconnect()
      chart.remove()

      chartRef.current = null
      candleSeriesRef.current = null
      volumeSeriesRef.current = null
      aiSeriesRef.current = null
      emaSeriesRef.current = null
      smaSeriesRef.current = null
      bollingerUpperRef.current = null
      bollingerLowerRef.current = null
    }
  }, [])


  useEffect(() => {
    const chart =
      chartRef.current

    const candleSeries =
      candleSeriesRef.current

    const volumeSeries =
      volumeSeriesRef.current

    if (
      !chart
      || !candleSeries
      || !volumeSeries
    ) {
      return
    }

    const candleData:
      CandlestickData[] =
      candles.map(
        (candle) => ({
          time: candle.time as UTCTimestamp,
          open: candle.open,
          high: candle.high,
          low: candle.low,
          close: candle.close,
        }),
      )

    const volumeData:
      HistogramData[] =
      candles.map(
        (candle) => ({
          time: candle.time as UTCTimestamp,
          value: candle.volume,
          color:
            candle.close
              >= candle.open
              ? 'rgba(34,197,94,0.35)'
              : 'rgba(239,68,68,0.35)',
        }),
      )

    candleSeries.setData(
      candleData,
    )

    volumeSeries.setData(
      volumeData,
    )

    emaSeriesRef.current
      ?.setData(
        computedIndicators.ema20,
      )

    smaSeriesRef.current
      ?.setData(
        computedIndicators.sma20,
      )

    bollingerUpperRef.current
      ?.setData(
        computedIndicators.upper,
      )

    bollingerLowerRef.current
      ?.setData(
        computedIndicators.lower,
      )

    if (
      lastTimeframeRef.current
      !== timeframe
    ) {
      chart.timeScale()
        .fitContent()

      lastTimeframeRef.current =
        timeframe
    }
  }, [
    candles,
    timeframe,
    computedIndicators,
  ])


  useEffect(() => {
    const aiSeries =
      aiSeriesRef.current

    if (!aiSeries) {
      return
    }

    const data:
      LineData[] =
      aiPoints.map(
        (point) => ({
          time: point.time as UTCTimestamp,
          value:
            point.score,
        }),
      )

    aiSeries.setData(
      data,
    )
  }, [aiPoints])


  useEffect(() => {
    volumeSeriesRef.current
      ?.applyOptions({
        visible:
          indicators.volume,
      })

    aiSeriesRef.current
      ?.applyOptions({
        visible:
          indicators.ai,
      })

    emaSeriesRef.current
      ?.applyOptions({
        visible:
          indicators.ema20,
      })

    smaSeriesRef.current
      ?.applyOptions({
        visible:
          indicators.sma20,
      })

    bollingerUpperRef.current
      ?.applyOptions({
        visible:
          indicators.bollinger,
      })

    bollingerLowerRef.current
      ?.applyOptions({
        visible:
          indicators.bollinger,
      })
  }, [indicators])


  const toggleIndicator = (
    key: IndicatorKey,
  ) => {
    setIndicators(
      (current) => ({
        ...current,
        [key]:
          !current[key],
      }),
    )
  }


  const latest =
    candles.length > 0
      ? candles[
          candles.length - 1
        ]
      : null

  const latestAI =
    aiPoints.length > 0
      ? aiPoints[
          aiPoints.length - 1
        ]
      : null


  return (
    <section className="rounded-2xl border border-[#d4af37]/15 bg-[#0b0b0a] p-6 shadow-[0_18px_60px_rgba(0,0,0,0.35)]">
      <div className="flex flex-col gap-5 border-b border-white/5 pb-5 xl:flex-row xl:items-end xl:justify-between">
        <div>
          <p className="text-xs uppercase tracking-[0.2em] text-white/30">
            Live Market Chart
          </p>

          <div className="mt-2 flex flex-wrap items-baseline gap-3">
            <h2 className="text-lg font-semibold">
              BTC / USDT
            </h2>

            {latest && (
              <span className="text-sm text-white/40">
                ${
                  latest.close
                    .toLocaleString(
                      'en-US',
                      {
                        minimumFractionDigits:
                          2,
                        maximumFractionDigits:
                          2,
                      },
                    )
                }
              </span>
            )}

            <span className="rounded-md border border-emerald-400/15 bg-emerald-400/[0.06] px-2 py-1 text-[10px] uppercase tracking-wider text-emerald-300/70">
              live
            </span>
          </div>

          <p className="mt-2 text-xs text-white/25">
            Candles are live. Add overlays and market tools from the Indicators menu.
          </p>

          {latestAI && (
            <div className="mt-3 flex flex-wrap items-center gap-3 text-xs">
              <span className="rounded-md border border-blue-400/20 bg-blue-400/[0.07] px-2.5 py-1 text-blue-200">
                AI score {latestAI.score >= 0 ? '+' : ''}
                {latestAI.score.toFixed(3)}
              </span>

              <span className="text-white/30">
                UP {(latestAI.up * 100).toFixed(1)}%
              </span>

              <span className="text-white/30">
                DOWN {(latestAI.down * 100).toFixed(1)}%
              </span>

              <span className="text-white/30">
                {latestAI.signal}
              </span>
            </div>
          )}
        </div>

        <div className="flex flex-col gap-3 xl:items-end">
          <div className="flex flex-wrap gap-2">
            <div className="relative">
              <button
                type="button"
                onClick={() =>
                  setIndicatorMenuOpen(
                    (value) =>
                      !value,
                  )
                }
                className="rounded-lg border border-[#d4af37]/20 bg-[#d4af37]/[0.04] px-3 py-2 text-xs font-medium text-[#e8d9ad] transition hover:border-[#d4af37]/35 hover:bg-[#d4af37]/[0.08]"
              >
                Indicators
              </button>

              {indicatorMenuOpen && (
                <div className="absolute right-0 z-20 mt-2 w-64 rounded-xl border border-[#d4af37]/20 bg-[#0b0b0a]/95 p-2 shadow-2xl shadow-black/60 backdrop-blur">
                  {([
                    ['ema20', 'EMA 20'],
                    ['sma20', 'SMA 20'],
                    ['bollinger', 'Bollinger Bands'],
                    ['volume', 'Volume'],
                    ['ai', 'AI Score'],
                  ] as Array<
                    [
                      IndicatorKey,
                      string,
                    ]
                  >).map(
                    ([
                      key,
                      label,
                    ]) => (
                      <button
                        key={key}
                        type="button"
                        onClick={() =>
                          toggleIndicator(
                            key,
                          )
                        }
                        className="flex w-full items-center justify-between rounded-lg px-3 py-2.5 text-left text-sm text-white/70 transition hover:bg-[#d4af37]/[0.055] hover:text-[#f0cf68]"
                      >
                        <span>
                          {label}
                        </span>

                        <span
                          className={[
                            'h-2.5 w-2.5 rounded-full border',
                            indicators[key]
                              ? 'border-[#f0cf68] bg-[#d4af37]'
                              : 'border-white/20',
                          ].join(' ')}
                        />
                      </button>
                    ),
                  )}
                </div>
              )}
            </div>

            <button
              type="button"
              onClick={() =>
                chartRef.current
                  ?.timeScale()
                  .fitContent()
              }
              className="rounded-lg border border-[#d4af37]/15 bg-black/20 px-3 py-2 text-xs text-white/45 transition hover:border-[#d4af37]/30 hover:text-[#f0cf68]"
            >
              Reset view
            </button>
          </div>

          <div>
            <p className="mb-2 text-[10px] uppercase tracking-[0.18em] text-white/25 xl:text-right">
              Chart timeframe
            </p>

            <div className="flex flex-wrap gap-1.5 xl:justify-end">
            {CHART_TIMEFRAMES.map(
              (item) => (
                <button
                  key={item}
                  type="button"
                  onClick={() =>
                    onTimeframeChange(
                      item,
                    )
                  }
                  className={[
                    'rounded-md border px-2.5 py-1.5 text-xs transition',
                    item === timeframe
                      ? 'border-white/30 bg-white text-black'
                      : 'border-white/10 bg-white/[0.02] text-white/45 hover:bg-white/[0.06] hover:text-white',
                  ].join(' ')}
                >
                  {item}
                </button>
              ),
            )}
            </div>
          </div>
        </div>
      </div>

      <div className="mt-4 flex flex-wrap gap-2 text-[11px]">
        {indicators.ema20 && (
          <span className="rounded-full border border-[#d89c45]/20 bg-[#d89c45]/[0.05] px-2.5 py-1 text-[#d9b36d]">
            EMA 20
          </span>
        )}

        {indicators.sma20 && (
          <span className="rounded-full border border-[#e7c85e]/20 bg-[#e7c85e]/[0.05] px-2.5 py-1 text-[#e7c85e]">
            SMA 20
          </span>
        )}

        {indicators.bollinger && (
          <span className="rounded-full border border-[#d5be7d]/20 bg-[#d5be7d]/[0.04] px-2.5 py-1 text-[#d5be7d]">
            Bollinger 20,2
          </span>
        )}
      </div>

      <div className="relative mt-4 min-h-[520px] overflow-hidden rounded-xl border border-[#d4af37]/10 bg-[#080807]">
        <div
          ref={chartContainerRef}
          className="w-full"
        />

        {loading
          && candles.length === 0 && (
          <div className="absolute inset-0 flex items-center justify-center bg-[#080d18]/70 text-sm text-white/35 backdrop-blur-sm">
            Loading live candles...
          </div>
        )}

        {error && (
          <div className="absolute left-4 top-4 rounded-lg border border-amber-400/20 bg-[#111827]/95 px-4 py-3 text-xs text-amber-200">
            {error}
          </div>
        )}
      </div>

      <div className="mt-4 flex flex-wrap items-center justify-between gap-3 border-t border-white/5 pt-4 text-xs text-white/25">
        <span>
          Pan · zoom · crosshair · indicators · reset view
        </span>

        <span>
          Candles refresh every 5s · AI overlay updates on completed 10m model candles.
        </span>
      </div>
    </section>
  )
}


export default BTCChart
