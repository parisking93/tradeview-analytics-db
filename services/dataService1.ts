import { Candle, Trade, PivotLevel, PivotType, DataResponse } from '../types';

const API_BASE = "/api";

const generatePivots = (currentPrice: number, symbol: string): PivotLevel[] => {
  const spread = currentPrice * 0.05;
  return [
    { id: 1, symbol, price: currentPrice + spread, type: PivotType.RESISTANCE, label: 'R1', color: '#ef5350' },
    { id: 2, symbol, price: currentPrice + (spread * 2), type: PivotType.RESISTANCE, label: 'R2', color: '#c62828' },
    { id: 3, symbol, price: currentPrice - spread, type: PivotType.SUPPORT, label: 'S1', color: '#26a69a' },
    { id: 4, symbol, price: currentPrice - (spread * 2), type: PivotType.SUPPORT, label: 'S2', color: '#00695c' },
  ];
};

export const searchPairs = async (query: string): Promise<string[]> => {
  if (!query) return [];
  const r = await fetch(`${API_BASE}/pairs?query=${encodeURIComponent(query)}`);
  if (!r.ok) return [];
  return await r.json();
};

export const fetchData = async (symbol: string): Promise<DataResponse> => {
  // puoi cambiare timeframe qui o farlo diventare parametro
  const timeframe = "5m";
  const r = await fetch(`${API_BASE}/market/${encodeURIComponent(symbol)}?timeframe=${timeframe}&limit=800`);
  if (!r.ok) throw new Error("API /market failed");

  const rows = await r.json();

  const candles: Candle[] = rows.map((row: any) => ({
    time: row.dt,          // datetime dal DB
    open: row.open,
    high: row.high,
    low: row.low,
    close: row.close
  }));

  const currentPrice = candles.length ? candles[candles.length - 1].close : 0;
  const pivots = generatePivots(currentPrice, symbol);

  // Trades reali: quando avrai tabella ordini, li prenderemo da /api/portfolio
  const trades: Trade[] = [];

  // Compatibilità con ChartComponent: se prima usavi ema_fast/slow
  const indicators = {
    ema_fast: rows.map((r: any) => ({ time: r.dt, value: r.ema })),
    ema_slow: rows.map((r: any) => ({ time: r.dt, value: r.ema })),
  };

  return { candles, trades, pivots, indicators };
};

export const fetchPortfolio = async (): Promise<Trade[]> => {
  const r = await fetch(`${API_BASE}/portfolio`);
  if (!r.ok) return [];
  return await r.json();
};
