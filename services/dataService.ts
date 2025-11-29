// --- START OF FILE dataService.ts ---

import { Candle, DataResponse, ForecastData, PivotLevel, PivotType, Trade, TradeType } from '../types';

const API_BASE = '/api';

// FIX: Funzione toUnix più robusta per date SQL 'YYYY-MM-DD HH:MM:SS'
const toUnix = (value: any): number | undefined => {
  if (value === undefined || value === null) return undefined;
  if (typeof value === 'number') {
    // Se è già un timestamp in millisecondi (es. 1700000000000), convertilo in secondi
    return value > 1e12 ? Math.floor(value / 1000) : value;
  }

  // Se è stringa, assicuriamoci che sia ISO-8601 compatibile (spazio -> T)
  const stringVal = String(value).replace(' ', 'T');
  // Aggiungiamo 'Z' se manca per forzare UTC, altrimenti il browser usa il fuso locale
  const finalString = stringVal.includes('Z') ? stringVal : stringVal + 'Z';

  const date = new Date(finalString);
  if (Number.isNaN(date.getTime())) return undefined;
  return Math.floor(date.getTime() / 1000);
};

const generatePivots = (
  candles: Candle[],
  symbol: string
): PivotLevel[] => {
  if (!candles || candles.length < 3) {
    return [
      { id: 1, symbol, price: 0, type: PivotType.RESISTANCE, label: 'R1', color: '#ff7f50' },
      { id: 2, symbol, price: 0, type: PivotType.RESISTANCE, label: 'R2', color: '#ff8fb1' },
      { id: 3, symbol, price: 0, type: PivotType.SUPPORT, label: 'S1', color: '#3dd5f3' },
      { id: 4, symbol, price: 0, type: PivotType.SUPPORT, label: 'S2', color: '#5eead4' },
    ];
  }

  const recent = candles.slice(-50);
  const highs = recent.map(c => c.high);
  const lows = recent.map(c => c.low);
  const closes = recent.map(c => c.close);
  const H = Math.max(...highs);
  const L = Math.min(...lows);
  const C = closes[closes.length - 1];
  const pivot = (H + L + C) / 3;
  const range = H - L;
  const R1 = 2 * pivot - L;
  const S1 = 2 * pivot - H;
  const R2 = pivot + range;
  const S2 = pivot - range;

  return [
    { id: 1, symbol, price: R1, type: PivotType.RESISTANCE, label: 'R1', color: '#ff7f50' },
    { id: 2, symbol, price: R2, type: PivotType.RESISTANCE, label: 'R2', color: '#ff8fb1' },
    { id: 3, symbol, price: S1, type: PivotType.SUPPORT, label: 'S1', color: '#3dd5f3' },
    { id: 4, symbol, price: S2, type: PivotType.SUPPORT, label: 'S2', color: '#5eead4' },
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
  const createdAt = raw.created_at ? new Date(raw.created_at).toISOString() : undefined;

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
    createdAt,
  };
};

export const searchPairs = async (query: string): Promise<string[]> => {
  if (!query) return [];
  const response = await fetch(`${API_BASE}/pairs?query=${encodeURIComponent(query)}`);
  if (!response.ok) return [];
  return await response.json();
};

const tfToApi = (tf?: string) => (tf ? tf.toLowerCase() : '4h');

export const fetchData = async (
  symbol: string,
  timeframe: string = '4h',
  endDate?: string,
  limit: number = 800
): Promise<DataResponse> => {

  const apiTimeframe = tfToApi(timeframe);
  const safeLimit = Math.min(limit, 12000);

  let marketUrl = `${API_BASE}/market/${encodeURIComponent(symbol)}?timeframe=${apiTimeframe}&limit=${safeLimit}`;
  if (endDate) {
    marketUrl += `&to=${encodeURIComponent(endDate)}`;
  }

  const [marketResponse, portfolioResponse] = await Promise.all([
    fetch(marketUrl),
    fetch(`${API_BASE}/portfolio`),
  ]);

  if (!marketResponse.ok) throw new Error('API /market failed');

  const marketPayload = await marketResponse.json();
  const candleSource = Array.isArray(marketPayload) ? marketPayload : marketPayload.candles || [];
  const forecastSource = !Array.isArray(marketPayload) && marketPayload.forecast ? marketPayload.forecast : [];

  const mappedCandles: Candle[] = candleSource
    .map((row: any) => {
      // Usiamo il toUnix migliorato
      const t = toUnix(row.time || row.timestamp || row.dt || row.ts);
      return {
        time: t as any,
        open: Number(row.open),
        high: Number(row.high),
        low: Number(row.low),
        close: Number(row.close),
      };
    })
    // Filtriamo rigorosamente i NaN o undefined
    .filter((c: Candle) => c.time !== undefined && !Number.isNaN(Number(c.time)) && !Number.isNaN(c.close));

  // Ordinamento e Deduplicazione essenziale per evitare crash del grafico
  const sortedCandles = mappedCandles.sort((a, b) => Number(a.time) - Number(b.time));
  const dedupCandles: Candle[] = [];
  const seenTimes = new Set<number>();
  for (const c of sortedCandles) {
    const t = Number(c.time);
    if (seenTimes.has(t)) continue;
    seenTimes.add(t);
    dedupCandles.push(c);
  }
  const candles = dedupCandles;

  const mappedForecast: ForecastData[] = (forecastSource as any[])
    .map((row: any) => ({
      time: row.time || row.timestamp || row.dt || row.ts,
      open: Number(row.open),
      high: Number(row.high),
      low: Number(row.low),
      close: Number(row.close),
    }))
    .filter((c: ForecastData) => {
      const ts = toUnix(c.time);
      return c.time !== undefined && ts !== undefined && !Number.isNaN(Number(c.close));
    });

  const sortedForecast = mappedForecast.sort((a, b) => Number(toUnix(a.time) || 0) - Number(toUnix(b.time) || 0));
  const dedupForecast: ForecastData[] = [];
  const seenForecastTimes = new Set<number>();
  for (const c of sortedForecast) {
    const t = Number(toUnix(c.time) || 0);
    if (seenForecastTimes.has(t)) continue;
    seenForecastTimes.add(t);
    dedupForecast.push(c);
  }

  const indicatorSource = !Array.isArray(marketPayload) && marketPayload.indicators ? marketPayload.indicators : {};
  const indicators = {
    ema_fast: (indicatorSource.ema_fast || []).map((item: any) => ({
      time: toUnix(item.time || item.timestamp || item.dt || item.ts),
      value: Number(item.value ?? item.ema_fast ?? item.ema),
    })).filter((i: any) => i.time),
    ema_slow: (indicatorSource.ema_slow || []).map((item: any) => ({
      time: toUnix(item.time || item.timestamp || item.dt || item.ts),
      value: Number(item.value ?? item.ema_slow ?? item.ema),
    })).filter((i: any) => i.time),
  };

  const currentPrice = endDate ? 0 :
    ((!Array.isArray(marketPayload) && marketPayload.lastPrice !== undefined
      ? Number(marketPayload.lastPrice)
      : undefined) ?? (candles.length ? candles[candles.length - 1].close : 0));

  const pivots = generatePivots(candles, symbol);

  const tradesPayload = portfolioResponse.ok ? await portfolioResponse.json() : [];
  const trades: Trade[] = Array.isArray(tradesPayload) ? tradesPayload.map(normalizeTrade) : [];

  return { candles, forecast: dedupForecast, trades, pivots, indicators };
};

export const fetchPortfolio = async (): Promise<Trade[]> => {
  const response = await fetch(`${API_BASE}/portfolio`);
  if (!response.ok) return [];
  const payload = await response.json();
  if (!Array.isArray(payload)) return [];
  return payload.map(normalizeTrade);
};
