import { Candle, Trade, PivotLevel, PivotType, DataResponse } from '../types';

const API_BASE = '/api';

const generatePivots = (currentPrice: number, symbol: string): PivotLevel[] => {
  const spread = currentPrice * 0.05;
  return [
    { id: 1, symbol, price: currentPrice + spread, type: PivotType.RESISTANCE, label: 'R1', color: '#ef5350' },
    { id: 2, symbol, price: currentPrice + spread * 2, type: PivotType.RESISTANCE, label: 'R2', color: '#c62828' },
    { id: 3, symbol, price: currentPrice - spread, type: PivotType.SUPPORT, label: 'S1', color: '#26a69a' },
    { id: 4, symbol, price: currentPrice - spread * 2, type: PivotType.SUPPORT, label: 'S2', color: '#00695c' },
  ];
};

export const searchPairs = async (query: string): Promise<string[]> => {
  if (!query) return [];
  const response = await fetch(`${API_BASE}/pairs?query=${encodeURIComponent(query)}`);
  if (!response.ok) return [];
  return await response.json();
};

export const fetchData = async (symbol: string): Promise<DataResponse> => {
  const timeframe = '5m';
  const response = await fetch(`${API_BASE}/market/${encodeURIComponent(symbol)}?timeframe=${timeframe}&limit=800`);
  if (!response.ok) throw new Error('API /market failed');

  const rows = await response.json();
  const candles: Candle[] = rows.map((row: any) => ({
    time: row.dt || row.timestamp || row.ts,
    open: Number(row.open),
    high: Number(row.high),
    low: Number(row.low),
    close: Number(row.close),
  }));

  const currentPrice = candles.length ? candles[candles.length - 1].close : 0;
  const pivots = generatePivots(currentPrice, symbol);

  const trades: Trade[] = [];

  const indicators = {
    ema_fast: rows.map((r: any) => ({ time: r.dt || r.timestamp || r.ts, value: Number(r.ema) })),
    ema_slow: rows.map((r: any) => ({ time: r.dt || r.timestamp || r.ts, value: Number(r.ema) })),
  };

  return { candles, trades, pivots, indicators };
};

export const fetchPortfolio = async (): Promise<Trade[]> => {
  const response = await fetch(`${API_BASE}/portfolio`);
  if (!response.ok) return [];
  return await response.json();
};
