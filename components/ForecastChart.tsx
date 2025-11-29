
import {
  AreaSeries,
  CandlestickSeries,
  ColorType,
  createChart,
  IChartApi,
  LineSeries,
  LineStyle,
  Time
} from 'lightweight-charts';
import React, { useEffect, useRef } from 'react';
import { CHART_BG_COLOR, CHART_GRID_COLOR, CHART_TEXT_COLOR } from '../constants';
import { Candle, Trade } from '../types';

interface ForecastChartProps {
  candles: Candle[];
  forecast: Candle[];
  activeTrade?: Trade;
  symbol: string;
  timeframe?: string;
  displayMode?: 'candles' | 'line';
  showBand?: boolean;
}

const toUnix = (value: any) => {
  if (value === undefined || value === null) return undefined;
  if (typeof value === 'number') {
    return value > 1e12 ? Math.floor(value / 1000) : value;
  }
  const stringVal = String(value).replace(' ', 'T');
  const finalString = stringVal.includes('Z') ? stringVal : stringVal + 'Z';
  const date = new Date(finalString);
  if (Number.isNaN(date.getTime())) return undefined;
  return Math.floor(date.getTime() / 1000);
};

const ForecastChart: React.FC<ForecastChartProps> = ({ candles, forecast, activeTrade, symbol, timeframe = '1D', displayMode = 'candles', showBand = true }) => {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const entryOverlayRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!chartContainerRef.current) return;

    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: CHART_BG_COLOR },
        textColor: CHART_TEXT_COLOR,
      },
      grid: {
        vertLines: { color: CHART_GRID_COLOR },
        horzLines: { color: CHART_GRID_COLOR },
      },
      width: chartContainerRef.current.clientWidth,
      height: chartContainerRef.current.clientHeight,
      timeScale: {
        borderColor: CHART_GRID_COLOR,
        timeVisible: true,
        secondsVisible: false,
      },
      rightPriceScale: {
        borderColor: CHART_GRID_COLOR,
      },
    });

    chartRef.current = chart;

    const marketSeries = displayMode === 'candles'
      ? chart.addSeries(CandlestickSeries, {
        upColor: '#60a5fa',
        downColor: '#1e3a8a',
        borderVisible: false,
        wickUpColor: '#60a5fa',
        wickDownColor: '#1e3a8a',
      })
      : chart.addSeries(LineSeries, {
        color: '#60a5fa',
        lineWidth: 2,
      });

    // Forecast series setup
    const forecastCloseSeries = displayMode === 'candles'
      ? chart.addSeries(CandlestickSeries, {
        upColor: '#fbbf24', // giallo
        downColor: '#a855f7', // viola
        borderVisible: false,
        wickUpColor: '#fbbf24',
        wickDownColor: '#a855f7',
      })
      : chart.addSeries(LineSeries, {
        color: '#fbbf24',
        lineWidth: 2,
      });

    // Extra lines + band only in line mode
    const forecastHighSeries = displayMode === 'line' && showBand
      ? chart.addSeries(LineSeries, { color: '#22c55e', lineWidth: 2, lineStyle: LineStyle.Solid })
      : null;
    const forecastLowSeries = displayMode === 'line' && showBand
      ? chart.addSeries(LineSeries, { color: '#16a34a', lineWidth: 2, lineStyle: LineStyle.Solid })
      : null;
    // Band fill (soft gradient) to hint the range without masking the chart
    const forecastBandFill = displayMode === 'line' && showBand
      ? chart.addSeries(AreaSeries, {
        topColor: 'rgba(34,197,94,0.05)',
        bottomColor: 'rgba(34,197,94,0.0)',
        lineColor: 'rgba(34,197,94,0)',
        lineWidth: 0,
        priceLineVisible: false,
        baseLineVisible: false,
      })
      : null;

    // Map data to Unix Timestamps
    const candleData = candles
      .map(d => {
        const t = toUnix(d.time);
        if (t === undefined) return null;
        return displayMode === 'candles'
          ? {
            time: t as Time,
            open: d.open,
            high: d.high,
            low: d.low,
            close: d.close
          }
          : {
            time: t as Time,
            value: d.close
          };
      })
      .filter(Boolean) as any[];
    marketSeries.setData(candleData);

    if (forecast.length > 0) {
      let minLow = Number.POSITIVE_INFINITY;
      const forecastData = forecast
        .map(f => {
          const t = toUnix(f.time);
          if (t === undefined) return null;
          if (displayMode === 'line') {
            minLow = Math.min(minLow, f.low ?? f.close);
          }
          return displayMode === 'candles'
            ? {
              time: t as Time,
              open: f.open,
              high: f.high,
              low: f.low,
              close: f.close
            }
            : {
              time: t as Time,
              value: f.close,
              high: f.high,
              low: f.low
            };
        })
        .filter(Boolean) as any[];
      forecastCloseSeries.setData(forecastData as any);

      if (displayMode === 'line' && forecastData.length) {
        const highs = (forecastData as any[]).map(d => ({ time: d.time, value: d.high ?? d.value }));
        const lows = (forecastData as any[]).map(d => ({ time: d.time, value: d.low ?? d.value }));
        const bandFillData = highs;

        if (forecastHighSeries) forecastHighSeries.setData(highs);
        if (forecastLowSeries) forecastLowSeries.setData(lows);
        const baseVal = Number.isFinite(minLow) ? minLow : 0;

        if (forecastBandFill) {
          forecastBandFill.applyOptions({ baseValue: { type: 'price', price: baseVal } });
          forecastBandFill.setData(bandFillData);
        }
      }
    }

    if (activeTrade) {
      marketSeries.createPriceLine({
        price: activeTrade.entryPrice,
        color: '#ffffff',
        lineWidth: 2,
        lineStyle: LineStyle.Solid,
        axisLabelVisible: true,
        title: 'ENTRY',
      });
      marketSeries.createPriceLine({
        price: activeTrade.takeProfit,
        color: '#4ade80',
        lineWidth: 2,
        lineStyle: LineStyle.Dotted,
        axisLabelVisible: true,
        title: 'TP',
      });
      marketSeries.createPriceLine({
        price: activeTrade.stopLoss,
        color: '#f87171',
        lineWidth: 2,
        lineStyle: LineStyle.Dotted,
        axisLabelVisible: true,
        title: 'SL',
      });
    }

    chart.timeScale().fitContent();

    const updateEntryOverlay = () => {
      if (!chartRef.current || !entryOverlayRef.current || !activeTrade?.createdAt) return;
      const time = toUnix(activeTrade.createdAt);
      if (time === undefined) return;
      const x = chartRef.current.timeScale().timeToCoordinate(time as Time);
      const y = chartRef.current.priceScale('right').priceToCoordinate(activeTrade.entryPrice);
      if (x === null || y === null || x === undefined || y === undefined) {
        entryOverlayRef.current.style.display = 'none';
        return;
      }
      entryOverlayRef.current.style.display = 'block';
      entryOverlayRef.current.style.top = `${y}px`;
      const dot = entryOverlayRef.current.querySelector('[data-entry-dot]') as HTMLDivElement | null;
      if (dot) {
        dot.style.left = `${x}px`;
        dot.style.top = `-4px`;
      }
    };

    updateEntryOverlay();

    const applySize = () => {
      if (chartContainerRef.current && chartRef.current) {
        const { clientWidth, clientHeight } = chartContainerRef.current;
        chartRef.current.applyOptions({ 
            width: clientWidth,
          height: clientHeight
        });
      }
    };

    applySize();

    const handleResize = () => applySize();
    window.addEventListener('resize', handleResize);

    const handleRange = () => updateEntryOverlay();
    chart.timeScale().subscribeVisibleLogicalRangeChange(handleRange);

    const resizeObserver = new ResizeObserver(() => applySize());
    resizeObserver.observe(chartContainerRef.current);

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.timeScale().unsubscribeVisibleLogicalRangeChange(handleRange);
      resizeObserver.disconnect();
      chart.remove();
      chartRef.current = null;
    };
  }, [candles, forecast, activeTrade, displayMode, timeframe, showBand]);

  return (
    <div className="relative w-full h-full group">
        <div ref={chartContainerRef} className="w-full h-full" />
        <div
          ref={entryOverlayRef}
          className="absolute left-0 right-0 border-t border-dashed border-white/70 pointer-events-none hidden"
          style={{ height: '0px' }}
        >
          <div
            data-entry-dot
            className="absolute w-2 h-2 -translate-x-1/2 -translate-y-1/2 rounded-full bg-white shadow-[0_0_8px_rgba(255,255,255,0.6)]"
            style={{ left: '0px', top: '0px' }}
          />
        </div>

        <div className="absolute top-4 left-4 z-10 bg-gray-900/90 backdrop-blur p-4 rounded-lg border border-gray-700 shadow-xl pointer-events-none">
            <h2 className="text-xl font-bold text-white mb-1">{symbol} <span className="text-blue-400 text-sm font-normal">AI Forecast</span></h2>
            <p className="text-xs text-gray-400 mb-2">{timeframe} Timeframe (forecast overlay {timeframe}+1)</p>
            <div className="flex flex-col gap-2 mt-2 text-xs font-mono">
          <div className="flex items-center gap-2">
            <div className="w-4 h-0.5 bg-blue-300"></div>
            <span className="text-gray-300">Market {displayMode === 'candles' ? 'candles' : 'line (close)'} </span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-0.5 bg-amber-300"></div>
            <span className="text-gray-300">Forecast {displayMode === 'candles' ? 'candles' : 'line (close)'} ({timeframe}+1)</span>
          </div>
          {displayMode === 'line' && (
            <div className="flex items-center gap-2">
              <div className="w-4 h-0.5 bg-green-400"></div>
              <div className="w-4 h-0.5 bg-emerald-500"></div>
              <span className="text-gray-300">Forecast band (P90 / P10)</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ForecastChart;
