
import React, { useEffect, useRef } from 'react';
import {
  createChart, 
  ColorType, 
  IChartApi, 
  Time, 
  CandlestickSeries,
  LineStyle
} from 'lightweight-charts';
import { Candle, Trade } from '../types';
import { CHART_BG_COLOR, CHART_GRID_COLOR, CHART_TEXT_COLOR } from '../constants';

interface ForecastChartProps {
  candles: Candle[];
  forecast: Candle[];
  activeTrade?: Trade; 
  symbol: string;
  timeframe?: string;
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

const ForecastChart: React.FC<ForecastChartProps> = ({ candles, forecast, activeTrade, symbol, timeframe = '1D' }) => {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

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

    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: '#60a5fa', 
      downColor: '#1e3a8a',
      borderVisible: false,
      wickUpColor: '#60a5fa',
      wickDownColor: '#1e3a8a',
    });

    const forecastSeries = chart.addSeries(CandlestickSeries, {
      upColor: '#fbbf24', // giallo
      downColor: '#a855f7', // viola
      borderVisible: false,
      wickUpColor: '#fbbf24',
      wickDownColor: '#a855f7',
    });
    
    // Map data to Unix Timestamps
    const candleData = candles
      .map(d => {
        const t = toUnix(d.time);
        if (t === undefined) return null;
        return {
          time: t as Time,
          open: d.open,
          high: d.high,
          low: d.low,
          close: d.close
        };
      })
      .filter(Boolean) as any[];
    candleSeries.setData(candleData);

    if (forecast.length > 0) {
      const forecastData = forecast
        .map(f => {
          const t = toUnix(f.time);
          if (t === undefined) return null;
          return {
            time: t as Time,
            open: f.open,
            high: f.high,
            low: f.low,
            close: f.close
          };
        })
        .filter(Boolean) as any[];
      forecastSeries.setData(forecastData);
    }

    if (activeTrade) {
        candleSeries.createPriceLine({
            price: activeTrade.entryPrice,
            color: '#ffffff',
            lineWidth: 2,
            lineStyle: LineStyle.Solid,
            axisLabelVisible: true,
            title: 'ENTRY',
        });
        candleSeries.createPriceLine({
            price: activeTrade.takeProfit,
            color: '#4ade80',
            lineWidth: 2,
            lineStyle: LineStyle.Dotted,
            axisLabelVisible: true,
            title: 'TP',
        });
        candleSeries.createPriceLine({
            price: activeTrade.stopLoss,
            color: '#f87171',
            lineWidth: 2,
            lineStyle: LineStyle.Dotted,
            axisLabelVisible: true,
            title: 'SL',
        });
    }

    chart.timeScale().fitContent();

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

    const resizeObserver = new ResizeObserver(() => applySize());
    resizeObserver.observe(chartContainerRef.current);

    return () => {
      window.removeEventListener('resize', handleResize);
      resizeObserver.disconnect();
      chart.remove();
      chartRef.current = null;
    };
  }, [candles, forecast, activeTrade]);

  return (
    <div className="relative w-full h-full group">
        <div ref={chartContainerRef} className="w-full h-full" />
        
        <div className="absolute top-4 left-4 z-10 bg-gray-900/90 backdrop-blur p-4 rounded-lg border border-gray-700 shadow-xl pointer-events-none">
            <h2 className="text-xl font-bold text-white mb-1">{symbol} <span className="text-blue-400 text-sm font-normal">AI Forecast</span></h2>
            <p className="text-xs text-gray-400 mb-2">{timeframe} Timeframe (forecast overlay {timeframe}+1)</p>
            <div className="flex flex-col gap-2 mt-2 text-xs font-mono">
              <div className="flex items-center gap-2">
                  <div className="w-4 h-0.5 bg-blue-300"></div>
                  <span className="text-gray-300">Market candles</span>
              </div>
              <div className="flex items-center gap-2">
                  <div className="w-4 h-0.5 bg-amber-300"></div>
                  <div className="w-4 h-0.5 bg-purple-400"></div>
                  <span className="text-gray-300">Forecast candles ({timeframe}+1)</span>
              </div>
            </div>
        </div>
    </div>
  );
};

export default ForecastChart;
