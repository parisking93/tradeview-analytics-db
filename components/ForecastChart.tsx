
import React, { useEffect, useRef } from 'react';
import { 
  createChart, 
  ColorType, 
  IChartApi, 
  Time, 
  CandlestickSeries,
  LineStyle,
  LineSeries
} from 'lightweight-charts';
import { Candle, Trade, ForecastData } from '../types';
import { CHART_BG_COLOR, CHART_GRID_COLOR, CHART_TEXT_COLOR } from '../constants';

interface ForecastChartProps {
  candles: Candle[];
  forecast: ForecastData[];
  activeTrade?: Trade; 
  symbol: string;
  timeframe?: string;
}

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
    
    // Map data to Unix Timestamps
    const candleData = candles.map(d => ({
        time: (new Date(d.time).getTime() / 1000) as Time,
        open: d.open,
        high: d.high,
        low: d.low,
        close: d.close
    }));
    candleSeries.setData(candleData);

    const p90Series = chart.addSeries(LineSeries, {
        color: 'rgba(239, 68, 68, 0.5)',
        lineWidth: 2,
        lineStyle: LineStyle.Solid,
        title: 'P90 (High)'
    });
    
    const p50Series = chart.addSeries(LineSeries, {
        color: '#fbbf24',
        lineWidth: 3,
        lineStyle: LineStyle.Dashed,
        title: 'P50 (Median)'
    });

    const p10Series = chart.addSeries(LineSeries, {
        color: 'rgba(239, 68, 68, 0.5)', 
        lineWidth: 2,
        lineStyle: LineStyle.Solid,
        title: 'P10 (Low)'
    });

    if (forecast.length > 0) {
        // Map forecast time to unix
        const mapForecast = (f: ForecastData, val: number) => ({
            time: (new Date(f.time).getTime() / 1000) as Time,
            value: val
        });

        p90Series.setData(forecast.map(f => mapForecast(f, f.p90)));
        p50Series.setData(forecast.map(f => mapForecast(f, f.p50)));
        p10Series.setData(forecast.map(f => mapForecast(f, f.p10)));
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

    const handleResize = () => {
      if (chartContainerRef.current && chartRef.current) {
        chartRef.current.applyOptions({ 
            width: chartContainerRef.current.clientWidth,
            height: chartContainerRef.current.clientHeight 
        });
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
      chartRef.current = null;
    };
  }, [candles, forecast, activeTrade]);

  return (
    <div className="relative w-full h-full group">
        <div ref={chartContainerRef} className="w-full h-full" />
        
        <div className="absolute top-4 left-4 z-10 bg-gray-900/90 backdrop-blur p-4 rounded-lg border border-gray-700 shadow-xl pointer-events-none">
            <h2 className="text-xl font-bold text-white mb-1">{symbol} <span className="text-blue-400 text-sm font-normal">AI Forecast</span></h2>
            <p className="text-xs text-gray-400 mb-2">{timeframe} Timeframe</p>
            <div className="flex flex-col gap-2 mt-2 text-xs font-mono">
                <div className="flex items-center gap-2">
                    <div className="w-4 h-0.5 bg-amber-400 border-b-2 border-dashed border-amber-400"></div>
                    <span className="text-gray-300">P50 (Median)</span>
                </div>
                <div className="flex items-center gap-2">
                    <div className="w-4 h-4 bg-red-500/20 border border-red-500/50 rounded flex items-center justify-center">
                         <div className="w-full h-0.5 bg-red-500/50"></div>
                    </div>
                    <span className="text-gray-300">P10 - P90 Range</span>
                </div>
            </div>
        </div>
    </div>
  );
};

export default ForecastChart;
