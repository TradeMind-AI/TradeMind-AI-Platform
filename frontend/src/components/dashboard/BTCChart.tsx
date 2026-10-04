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
  useRef,
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

type AISeriesHandle = {
  setData: (
    data: LineData[],
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


const BTCChart = ({
  candles,
  timeframe,
  onTimeframeChange,
  loading = false,
  error = null,
  aiPoints,
}: BTCChartProps) => {
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
    useRef<AISeriesHandle | null>(
      null,
    )

  const lastTimeframeRef =
    useRef<ChartTimeframe | null>(
      null,
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
        height: 460,

        layout: {
          background: {
            color: '#080d18',
          },
          textColor:
            'rgba(255,255,255,0.55)',
        },

        grid: {
          vertLines: {
            color:
              'rgba(255,255,255,0.04)',
          },
          horzLines: {
            color:
              'rgba(255,255,255,0.04)',
          },
        },

        crosshair: {
          vertLine: {
            color:
              'rgba(255,255,255,0.20)',
            labelBackgroundColor:
              '#111827',
          },
          horzLine: {
            color:
              'rgba(255,255,255,0.20)',
            labelBackgroundColor:
              '#111827',
          },
        },

        rightPriceScale: {
          borderColor:
            'rgba(255,255,255,0.08)',
        },

        timeScale: {
          borderColor:
            'rgba(255,255,255,0.08)',
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
          color: '#60a5fa',
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
    <section className="rounded-2xl border border-white/10 bg-white/[0.025] p-6">
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
            Candles are live. The blue overlay is the frozen 10m AI directional score.
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

        <div>
          <p className="mb-2 text-[10px] uppercase tracking-[0.18em] text-white/25">
            Chart timeframe
          </p>

          <div className="flex flex-wrap gap-1.5">
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

      <div className="relative mt-5 min-h-[460px]">
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
          Pan and zoom with the mouse or trackpad.
        </span>

        <span>
          Candles refresh every 5s · AI overlay updates on completed 10m model candles.
        </span>
      </div>
    </section>
  )
}


export default BTCChart
