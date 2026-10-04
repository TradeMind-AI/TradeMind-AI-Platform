import {
  CandlestickSeries,
  HistogramSeries,
  createChart,
  type CandlestickData,
  type HistogramData,
  type UTCTimestamp,
} from 'lightweight-charts'

import {
  useEffect,
  useRef,
} from 'react'


export type BTCCandle = {
  time: number
  open: number
  high: number
  low: number
  close: number
  volume: number
}


type BTCChartProps = {
  candles: BTCCandle[]
}


const BTCChart = ({
  candles,
}: BTCChartProps) => {
  const chartContainerRef =
    useRef<HTMLDivElement | null>(null)


  useEffect(() => {
    if (!chartContainerRef.current) {
      return
    }

    if (candles.length === 0) {
      return
    }

    const container =
      chartContainerRef.current

    const chart = createChart(
      container,
      {
        width: container.clientWidth,
        height: 420,

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

        rightPriceScale: {
          borderColor:
            'rgba(255,255,255,0.08)',
        },

        timeScale: {
          borderColor:
            'rgba(255,255,255,0.08)',
          timeVisible: true,
          secondsVisible: false,
        },
      },
    )


    const candleSeries =
      chart.addSeries(
        CandlestickSeries,
        {
          upColor: '#22c55e',
          downColor: '#ef4444',

          borderVisible: false,

          wickUpColor: '#22c55e',
          wickDownColor: '#ef4444',
        },
      )


    const candleData: CandlestickData[] =
      candles.map((candle) => ({
        time: candle.time as UTCTimestamp,
        open: candle.open,
        high: candle.high,
        low: candle.low,
        close: candle.close,
      }))


    candleSeries.setData(
      candleData,
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


    const volumeData: HistogramData[] =
      candles.map((candle) => ({
        time: candle.time as UTCTimestamp,
        value: candle.volume,

        color:
          candle.close >= candle.open
            ? 'rgba(34,197,94,0.35)'
            : 'rgba(239,68,68,0.35)',
      }))


    volumeSeries.setData(
      volumeData,
    )


    chart.timeScale().fitContent()


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
    }
  }, [candles])


  return (
    <section className="rounded-2xl border border-white/10 bg-white/[0.025] p-6">
      <div className="mb-5 flex items-center justify-between">
        <div>
          <p className="text-xs uppercase tracking-[0.2em] text-white/30">
            Market
          </p>

          <h2 className="mt-2 text-lg font-semibold">
            BTC / USDT
          </h2>
        </div>

        <span className="rounded-md bg-white/5 px-3 py-1.5 text-xs text-white/40">
          1H
        </span>
      </div>

      <div
        ref={chartContainerRef}
        className="w-full"
      />
    </section>
  )
}


export default BTCChart