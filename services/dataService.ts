import { Candle, Trade, PivotLevel, PivotType, DataResponse, TradeType } from '../types';

const API_BASE = '/api';
const toUnix = (value: any): number | undefined => {
  if (value === undefined || value === null) return undefined;
  if (typeof value === 'number') return value > 1e12 ? Math.floor(value / 1000) : value;
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return undefined;
  return Math.floor(date.getTime() / 1000);
};

const generatePivots = (
  currentPrice: number,
  symbol: string,
  spreadHint?: number,
  atrHint?: number
): PivotLevel[] => {
  const stepCandidate = atrHint && atrHint > 0 ? atrHint : spreadHint && spreadHint > 0 ? spreadHint : currentPrice * 0.01;
  const step = stepCandidate > 0 ? stepCandidate : 1;
  return [
    { id: 1, symbol, price: currentPrice + step, type: PivotType.RESISTANCE, label: 'R1', color: '#ef5350' },
    { id: 2, symbol, price: currentPrice + step * 2, type: PivotType.RESISTANCE, label: 'R2', color: '#c62828' },
    { id: 3, symbol, price: currentPrice - step, type: PivotType.SUPPORT, label: 'S1', color: '#26a69a' },
    { id: 4, symbol, price: currentPrice - step * 2, type: PivotType.SUPPORT, label: 'S2', color: '#00695c' },
  ];
};

const normalizeTrade = (raw: any): Trade => {
  const statusRaw = String(raw.status || raw.subtype || '').toUpperCase();
  const status: 'OPEN' | 'CLOSED' | 'PENDING' =
    statusRaw === 'CLOSED' ? 'CLOSED' : statusRaw === 'PENDING' ? 'PENDING' : 'OPEN';

  const typeRaw = String(raw.type || raw.side || raw.position || '').toUpperCase();
  const type = typeRaw.includes('SELL') || typeRaw.includes('SHORT') ? TradeType.SHORT : TradeType.LONG;

  const symbol = raw.symbol || raw.pair || `${raw.base || ''}${raw.quote || ''}`;
  const entryDateSource = raw.entryDate || raw.record_date || raw.created_at;
  const entryDate = entryDateSource
    ? new Date(entryDateSource).toISOString().slice(0, 10)
    : '';

  const entryPrice = raw.entryPrice ?? raw.price_entry ?? raw.price_avg ?? raw.price ?? 0;

  return {
    id: Number(raw.id ?? raw.order_id ?? Date.now()),
    symbol,
    entryDate,
    type,
    entryPrice: Number(entryPrice),
    stopLoss: Number(raw.stopLoss ?? raw.stop_loss ?? 0),
    takeProfit: Number(raw.takeProfit ?? raw.take_profit ?? 0),
    status,
    pnl: raw.pnl !== null && raw.pnl !== undefined ? Number(raw.pnl) : undefined,
  };
};

export const searchPairs = async (query: string): Promise<string[]> => {
  if (!query) return [];
  const response = await fetch(`${API_BASE}/pairs?query=${encodeURIComponent(query)}`);
  if (!response.ok) return [];
  return await response.json();
};

export const fetchData = async (symbol: string): Promise<DataResponse> => {
  const timeframe = '4h';
  const [marketResponse, portfolioResponse] = await Promise.all([
    fetch(`${API_BASE}/market/${encodeURIComponent(symbol)}?timeframe=${timeframe}&limit=800`),
    fetch(`${API_BASE}/portfolio`),
  ]);

  if (!marketResponse.ok) throw new Error('API /market failed');

  const marketPayload = await marketResponse.json();
  const candleSource = Array.isArray(marketPayload) ? marketPayload : marketPayload.candles || [];

  const candles: Candle[] = candleSource
    .map((row: any) => ({
      time: toUnix(row.time || row.timestamp || row.dt || row.ts) ?? row.time ?? row.timestamp ?? row.dt ?? row.ts,
      open: Number(row.open),
      high: Number(row.high),
      low: Number(row.low),
      close: Number(row.close),
    }))
    .filter((c: Candle) => !Number.isNaN(c.close));

  const indicatorSource = !Array.isArray(marketPayload) && marketPayload.indicators ? marketPayload.indicators : {};
  const indicators = {
    ema_fast: (indicatorSource.ema_fast || []).map((item: any) => ({
      time: toUnix(item.time || item.timestamp || item.dt || item.ts) ?? item.time ?? item.timestamp ?? item.dt ?? item.ts,
      value: Number(item.value ?? item.ema_fast ?? item.ema),
    })),
    ema_slow: (indicatorSource.ema_slow || []).map((item: any) => ({
      time: toUnix(item.time || item.timestamp || item.dt || item.ts) ?? item.time ?? item.timestamp ?? item.dt ?? item.ts,
      value: Number(item.value ?? item.ema_slow ?? item.ema),
    })),
  };

  const lastCandle = candleSource.length ? candleSource[candleSource.length - 1] : undefined;
  const currentPrice =
    (!Array.isArray(marketPayload) && marketPayload.lastPrice !== undefined
      ? Number(marketPayload.lastPrice)
      : undefined) ?? (candles.length ? candles[candles.length - 1].close : 0);

  const atrHint = lastCandle && lastCandle.atr !== undefined ? Number(lastCandle.atr) : undefined;
  const spreadHint =
    !Array.isArray(marketPayload) && marketPayload.spread !== undefined
      ? Number(marketPayload.spread)
      : lastCandle && lastCandle.spread !== undefined
      ? Number(lastCandle.spread)
      : undefined;

  const pivots = generatePivots(currentPrice, symbol, spreadHint, atrHint);

  const tradesPayload = portfolioResponse.ok ? await portfolioResponse.json() : [];
  const trades: Trade[] = Array.isArray(tradesPayload) ? tradesPayload.map(normalizeTrade) : [];

  return { candles, trades, pivots, indicators };
};

export const fetchPortfolio = async (): Promise<Trade[]> => {
  const response = await fetch(`${API_BASE}/portfolio`);
  if (!response.ok) return [];
  const payload = await response.json();
  if (!Array.isArray(payload)) return [];
  return payload.map(normalizeTrade);
};
