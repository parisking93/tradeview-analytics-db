// --- START OF FILE ChartComponent.tsx ---

import {
  CandlestickSeries,
  ColorType,
  createChart,
  IChartApi,
  IPriceLine, // Importiamo l'interfaccia per le linee
  ISeriesApi,
  LineStyle,
  SeriesMarker,
  Time
} from 'lightweight-charts';
import React, { useEffect, useRef } from 'react';
import { CHART_BG_COLOR, CHART_GRID_COLOR, CHART_TEXT_COLOR, COLOR_LONG, COLOR_SHORT } from '../constants';
import { Candle, PivotLevel, PivotType, Trade, TradeType } from '../types';

interface ChartComponentProps {
  data: Candle[];
  trades: Trade[];
  pivots: PivotLevel[];
  symbol: string;
  timeframe?: string;
  onRequestMore?: (earliestTime: number) => void;
  isLoadingMore?: boolean;
}

const ChartComponent: React.FC<ChartComponentProps> = ({
  data,
  trades,
  pivots,
  symbol,
  timeframe,
  onRequestMore,
  isLoadingMore
}) => {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);

  // --- NUOVO REF: Tiene traccia delle linee attive per poterle cancellare ---
  const pivotLinesRef = useRef<IPriceLine[]>([]);

  const loadingRef = useRef(false);

  useEffect(() => {
    loadingRef.current = !!isLoadingMore;
  }, [isLoadingMore]);

  // Inizializzazione del Grafico
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

    const candlestickSeries = chart.addSeries(CandlestickSeries, {
      upColor: COLOR_LONG,
      downColor: COLOR_SHORT,
      borderVisible: false,
      wickUpColor: COLOR_LONG,
      wickDownColor: COLOR_SHORT,
    });
    seriesRef.current = candlestickSeries;

    const handleResize = () => {
      if (chartContainerRef.current && chartRef.current) {
        chartRef.current.applyOptions({
          width: chartContainerRef.current.clientWidth,
          height: chartContainerRef.current.clientHeight
        });
      }
    };
    window.addEventListener('resize', handleResize);

    const handleVisibleRangeChange = (newVisibleRange: any) => {
      if (loadingRef.current || !onRequestMore) return;

      // Triggera quando l'utente è molto vicino all'inizio
      if (newVisibleRange && newVisibleRange.from < 5) {
        if (data.length > 0) {
          const firstCandleTime = Number(data[0].time) * 1000;
          loadingRef.current = true;
          onRequestMore(firstCandleTime);
        }
      }
    };

    chart.timeScale().subscribeVisibleLogicalRangeChange(handleVisibleRangeChange);

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.timeScale().unsubscribeVisibleLogicalRangeChange(handleVisibleRangeChange);
      chart.remove();
      chartRef.current = null;
      seriesRef.current = null;
      pivotLinesRef.current = []; // Reset array alla distruzione
    };
  }, []);

  // Update dei Dati
  useEffect(() => {
    if (!seriesRef.current || !chartRef.current) return;

    // 1. Preparazione Dati
    const rawData = data.map(d => ({
      time: Number(d.time) as Time,
      open: d.open,
      high: d.high,
      low: d.low,
      close: d.close
    }));

    // 2. Ordinamento e Deduplicazione
    rawData.sort((a, b) => (a.time as number) - (b.time as number));

    const uniqueData = [];
    let lastTime = -1;
    for (const item of rawData) {
      const t = item.time as number;
      if (t > lastTime && !Number.isNaN(t)) {
        uniqueData.push(item);
        lastTime = t;
      }
    }

    // 3. Set Data
    try {
      seriesRef.current.setData(uniqueData);
    } catch (e) {
      console.error("Chart setData Error:", e);
    }

    // 4. Markers (Ordini)
    const markers: SeriesMarker<Time>[] = [];
    trades
      .filter(t => t.symbol?.replace('/', '') === symbol.replace('/', ''))
      .forEach(t => {
        const tString = t.entryDate.toString().replace(' ', 'T') + (t.entryDate.toString().includes('Z') ? '' : 'Z');
        const tStamp = Math.floor(new Date(tString).getTime() / 1000);

        if (!Number.isNaN(tStamp)) {
          markers.push({
            time: tStamp as Time,
            position: t.type === TradeType.LONG ? 'belowBar' : 'aboveBar',
            color: t.type === TradeType.LONG ? COLOR_LONG : COLOR_SHORT,
            shape: t.type === TradeType.LONG ? 'arrowUp' : 'arrowDown',
            text: `${t.type} @ ${t.entryPrice}`,
          });
        }
      });

    markers.sort((a, b) => (a.time as number) - (b.time as number));

    if (seriesRef.current && typeof seriesRef.current.setMarkers === 'function') {
      seriesRef.current.setMarkers(markers);
    }

    // 5. PIVOTS MANAGEMENT (FIXED)

    // A. Rimuovi le vecchie linee
    if (pivotLinesRef.current.length > 0) {
      pivotLinesRef.current.forEach(line => {
        seriesRef.current?.removePriceLine(line);
      });
      pivotLinesRef.current = []; // Svuota l'array
    }

    // B. Crea le nuove linee
    pivots.forEach(pivot => {
      const newLine = seriesRef.current?.createPriceLine({
        price: pivot.price,
        color: pivot.color || (pivot.type === PivotType.RESISTANCE ? COLOR_SHORT : COLOR_LONG),
        lineWidth: 2,
        lineStyle: LineStyle.Dashed,
        axisLabelVisible: true,
        title: pivot.label,
      });

      // Salva il riferimento alla nuova linea per poterla cancellare dopo
      if (newLine) {
        pivotLinesRef.current.push(newLine);
      }
    });

  }, [data, trades, pivots, symbol]);

  return (
    <div className="relative w-full h-full">
      <div ref={chartContainerRef} className="w-full h-full" />
      {isLoadingMore && (
        <div className="absolute top-10 left-1/2 transform -translate-x-1/2 bg-blue-600/80 backdrop-blur text-white text-xs px-3 py-1 rounded-full shadow-lg z-20 animate-pulse border border-blue-400">
          Caricamento storico...
        </div>
      )}
      <div className="absolute top-4 left-4 z-10 bg-gray-850/80 backdrop-blur p-2 rounded border border-gray-700 shadow-lg pointer-events-none">
        <h2 className="text-xl font-bold text-white">{symbol}</h2>
        <p className="text-xs text-gray-400">{timeframe}</p>
      </div>
    </div>
  );
};

export default ChartComponent;
