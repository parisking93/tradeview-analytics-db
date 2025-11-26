import React, { useEffect, useRef } from 'react';
import { 
  createChart,
  ColorType,
  IChartApi,
  SeriesMarker, 
  Time, 
  CandlestickSeries,
  LineStyle 
} from 'lightweight-charts';
import { Candle, Trade, PivotLevel, TradeType, PivotType } from '../types';
import { CHART_BG_COLOR, CHART_GRID_COLOR, CHART_TEXT_COLOR, COLOR_LONG, COLOR_SHORT } from '../constants';

interface ChartComponentProps {
  data: Candle[];
  trades: Trade[];
  pivots: PivotLevel[];
  symbol: string;
}

const ChartComponent: React.FC<ChartComponentProps> = ({ data, trades, pivots, symbol }) => {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  
  useEffect(() => {
    if (!chartContainerRef.current) return;

    // 1. Initialize Chart
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
      },
      rightPriceScale: {
        borderColor: CHART_GRID_COLOR,
      },
    });

    chartRef.current = chart;

    // 2. Add Candlestick Series
    // Note: In v5, we pass the Series Class (CandlestickSeries) as the first argument
    const candlestickSeries = chart.addSeries(CandlestickSeries, {
      upColor: COLOR_LONG,
      downColor: COLOR_SHORT,
      borderVisible: false,
      wickUpColor: COLOR_LONG,
      wickDownColor: COLOR_SHORT,
    });
    
    // 3. Set Data
    const chartData = data.map(d => ({
        time: d.time as Time,
        open: d.open,
        high: d.high,
        low: d.low,
        close: d.close
    }));
    
    candlestickSeries.setData(chartData);

    // 4. Add Pivot Lines (Resistance & Support)
    pivots.forEach(pivot => {
        candlestickSeries.createPriceLine({
            price: pivot.price,
            color: pivot.color || (pivot.type === PivotType.RESISTANCE ? COLOR_SHORT : COLOR_LONG),
            lineWidth: 2,
            lineStyle: LineStyle.Dashed,
            axisLabelVisible: true,
            title: pivot.label,
        });
    });

    // 5. Add Trade Markers (Entry Points)
    const markers: SeriesMarker<Time>[] = [];
    
    trades.forEach(trade => {
        // Add Entry Marker
        markers.push({
            time: trade.entryDate as Time,
            position: trade.type === TradeType.LONG ? 'belowBar' : 'aboveBar',
            color: trade.type === TradeType.LONG ? COLOR_LONG : COLOR_SHORT,
            shape: trade.type === TradeType.LONG ? 'arrowUp' : 'arrowDown',
            text: `${trade.type} @ ${trade.entryPrice}`,
        });

        // If the trade is OPEN, draw SL and TP lines using createPriceLine
        if (trade.status === 'OPEN') {
            candlestickSeries.createPriceLine({
                price: trade.takeProfit,
                color: '#4ade80', // Greenish
                lineWidth: 1,
                lineStyle: LineStyle.Solid,
                axisLabelVisible: true,
                title: `TP #${trade.id}`,
            });

             candlestickSeries.createPriceLine({
                price: trade.stopLoss,
                color: '#f87171', // Reddish
                lineWidth: 1,
                lineStyle: LineStyle.Solid,
                axisLabelVisible: true,
                title: `SL #${trade.id}`,
            });
        }
    });

    // IMPORTANT: Markers must be sorted by time ascending
    markers.sort((a, b) => {
        const timeA = a.time as string;
        const timeB = b.time as string;
        return timeA.localeCompare(timeB);
    });

    // Safety check: ensure series exists and setMarkers is a function before calling
    const seriesWithMarkers = candlestickSeries as unknown as { setMarkers?: (m: SeriesMarker<Time>[]) => void };
    if (seriesWithMarkers && typeof seriesWithMarkers.setMarkers === 'function') {
        seriesWithMarkers.setMarkers(markers);
    }

    // Fit content nicely
    if (chartData.length > 0) {
        chart.timeScale().fitContent();
    }

    // Handle Resize
    const handleResize = () => {
      if (chartContainerRef.current && chartRef.current) {
        chartRef.current.applyOptions({ 
            width: chartContainerRef.current.clientWidth,
            height: chartContainerRef.current.clientHeight 
        });
      }
    };

    window.addEventListener('resize', handleResize);

    // Cleanup
    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
      chartRef.current = null;
    };
  }, [data, trades, pivots]);

  return (
    <div className="relative w-full h-full">
        <div ref={chartContainerRef} className="w-full h-full" />
        <div className="absolute top-4 left-4 z-10 bg-gray-850/80 backdrop-blur p-2 rounded border border-gray-700 shadow-lg pointer-events-none">
            <h2 className="text-xl font-bold text-white">{symbol}</h2>
            <p className="text-xs text-gray-400">Daily Timeframe</p>
        </div>
    </div>
  );
};

export default ChartComponent;