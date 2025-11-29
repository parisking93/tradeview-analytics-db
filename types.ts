
// Defines the structure of a candlestick for the Lightweight Chart
export interface Candle {
  time: string; // Format: 'yyyy-mm-dd'
  open: number;
  high: number;
  low: number;
  close: number;
}

export enum TradeType {
  LONG = 'LONG',
  SHORT = 'SHORT',
}

// Defines the structure of a trade/order (Assumed to be in a separate table 'orders' in the DB)
export interface Trade {
  id: number;
  symbol: string; // Maps to 'pair' in currency table
  entryDate: string; // 'yyyy-mm-dd'
  createdAt?: string;
  type: TradeType;
  entryPrice: number;
  stopLoss: number;
  takeProfit: number;
  status: 'OPEN' | 'CLOSED' | 'PENDING';
  pnl?: number;
}

export enum PivotType {
  RESISTANCE = 'RESISTANCE',
  SUPPORT = 'SUPPORT',
  PIVOT = 'PIVOT'
}

export interface PivotLevel {
  id: number;
  symbol: string;
  price: number;
  type: PivotType;
  label: string;
  color?: string;
}

export interface ForecastData extends Candle {}

// --- DATABASE SCHEMA MAPPING ---

// Exact mapping of the 'currency' table from DBeaver
export interface CurrencyRecord {
  id: number;
  pair: string;
  quote: string;
  base: string;
  day: string; // Timestamp/Date
  price: number;
  range: string;
  interval_min: number;
  since: number;
  open: number;
  close: number;
  start_price: number;
  current_price: number;
  change_pct: number;
  direction: string;
  high: number;
  low: number;
  volume: number;
  volume_label: string;
  bid: number;
  ask: number;
  last: number;
  mid: number;
  spread: number;
  ema_fast: number;
  ema_slow: number;
  atr: number;
}

export interface PairLimitRecord {
  id: number;
  lot_decimals: number;
  ordermin: number;
  pair_decimals: number;
  fee_volume_currency: string;
  leverage_buy: number;
  leverage_sell: number;
  // ... other fields from pair_limits
}

export interface DataResponse {
  candles: Candle[];
  trades: Trade[];
  pivots: PivotLevel[];
  indicators?: {
    ema_fast: { time: string, value: number }[];
    ema_slow: { time: string, value: number }[];
  },
  forecast?: ForecastData[];
}
