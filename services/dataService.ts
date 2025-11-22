
import { Candle, Trade, PivotLevel, TradeType, PivotType, DataResponse, CurrencyRecord } from '../types';

/**
 * SIMULATION NOTE:
 * 
 * This service simulates a backend connecting to your MySQL 'trading' database.
 * 
 * Query Logic Simulated:
 * 1. Market Data: SELECT * FROM currency WHERE pair = ? ORDER BY day ASC
 * 2. Search: SELECT DISTINCT pair FROM currency WHERE pair LIKE ?
 * 3. Portfolio: SELECT * FROM orders WHERE status = 'OPEN' (Assumed table)
 */

// --- MOCK ORDERS TABLE (Assumed to exist alongside 'currency') ---
const MOCK_ORDERS_TABLE: Trade[] = [
    {
        id: 101,
        symbol: 'BTCUSD',
        entryDate: '2023-04-01',
        type: TradeType.LONG,
        entryPrice: 42000,
        stopLoss: 41000,
        takeProfit: 45000,
        status: 'CLOSED',
        pnl: 1200.50
    },
    {
        id: 102,
        symbol: 'BTCUSD',
        entryDate: '2023-04-10',
        type: TradeType.SHORT,
        entryPrice: 45500,
        stopLoss: 46000,
        takeProfit: 43000,
        status: 'OPEN',
    },
    {
        id: 103,
        symbol: 'ETHUSD',
        entryDate: '2023-04-12',
        type: TradeType.LONG,
        entryPrice: 2500,
        stopLoss: 2400,
        takeProfit: 2800,
        status: 'OPEN',
    },
    {
        id: 104,
        symbol: 'SOLUSD',
        entryDate: '2023-04-11',
        type: TradeType.LONG,
        entryPrice: 135.50,
        stopLoss: 128.00,
        takeProfit: 155.00,
        status: 'OPEN',
    },
     {
        id: 105,
        symbol: 'EURUSD',
        entryDate: '2023-04-05',
        type: TradeType.SHORT,
        entryPrice: 1.0950,
        stopLoss: 1.1000,
        takeProfit: 1.0800,
        status: 'PENDING', 
    }
];

// --- MOCK 'currency' TABLE GENERATOR ---
// Simulates the data stored in your MySQL table
const generateCurrencyTableData = (count: number, pair: string): CurrencyRecord[] => {
  const rows: CurrencyRecord[] = [];
  let time = new Date('2023-01-01').getTime();
  
  // Determine base price based on pair
  let price = 100;
  if (pair.includes('BTC')) price = 45000;
  else if (pair.includes('ETH')) price = 2800;
  else if (pair.includes('SOL')) price = 140;
  else if (pair.includes('EUR')) price = 1.08;
  else if (pair.includes('JPY')) price = 150;

  for (let i = 0; i < count; i++) {
    const open = price;
    const volatility = pair.includes('USD') && !pair.includes('BTC') && !pair.includes('ETH') ? 0.005 : (price * 0.02); 
    
    const high = open + Math.random() * volatility;
    const low = open - Math.random() * volatility;
    const close = low + Math.random() * (high - low);
    
    // Simulate Indicators stored in DB
    const ema_fast = close * 0.99;
    const ema_slow = close * 0.95;
    const atr = volatility * 0.8;

    rows.push({
      id: i + 1,
      pair: pair,
      quote: 'USD', // Simplified
      base: pair.replace('USD', ''),
      day: new Date(time).toISOString().split('T')[0], // Matches 'day' column type
      price: close,
      range: '1D',
      interval_min: 1440,
      since: time,
      open: Number(open.toFixed(2)),
      close: Number(close.toFixed(2)),
      start_price: open, // Assuming start_price ~ open
      current_price: close,
      change_pct: ((close - open) / open) * 100,
      direction: close >= open ? 'UP' : 'DOWN',
      high: Number(high.toFixed(2)),
      low: Number(low.toFixed(2)),
      volume: Math.floor(Math.random() * 10000),
      volume_label: 'K',
      bid: close - 0.1,
      ask: close + 0.1,
      last: close,
      mid: close,
      spread: 0.2,
      ema_fast: Number(ema_fast.toFixed(2)),
      ema_slow: Number(ema_slow.toFixed(2)),
      atr: Number(atr.toFixed(4))
    });

    price = close;
    time += 86400000; // +1 Day
  }
  return rows;
};

const getTradesForSymbol = (symbol: string, currencyRows: CurrencyRecord[]): Trade[] => {
  // Filter trades from our assumed "orders" table
  const dbTrades = MOCK_ORDERS_TABLE.filter(t => t.symbol === symbol);
  
  // Align dates for visualization purposes
  return dbTrades.map((trade, index) => {
      const rowIndex = currencyRows.length - (5 + index * 5);
      if (rowIndex >= 0) {
          return { ...trade, entryDate: currencyRows[rowIndex].day };
      }
      return trade;
  });
};

const generatePivots = (currentPrice: number, symbol: string): PivotLevel[] => {
  const spread = currentPrice * 0.05;
  return [
    { id: 1, symbol: symbol, price: currentPrice + spread, type: PivotType.RESISTANCE, label: 'R1', color: '#ef5350' },
    { id: 2, symbol: symbol, price: currentPrice + (spread * 2), type: PivotType.RESISTANCE, label: 'R2', color: '#c62828' },
    { id: 3, symbol: symbol, price: currentPrice - spread, type: PivotType.SUPPORT, label: 'S1', color: '#26a69a' },
    { id: 4, symbol: symbol, price: currentPrice - (spread * 2), type: PivotType.SUPPORT, label: 'S2', color: '#00695c' },
  ];
};

// --- Main Service Functions ---

export const searchPairs = async (query: string): Promise<string[]> => {
    // Simulates: SELECT DISTINCT pair FROM currency WHERE pair LIKE '%query%'
    await new Promise(resolve => setTimeout(resolve, 300)); 
    
    const dbPairs = [
        'BTCUSD', 'ETHUSD', 'SOLUSD', 'XRPUSD', 
        'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD',
        'AAPL', 'TSLA', 'NVDA', 'MSFT'
    ];

    if (!query) return [];
    return dbPairs.filter(p => p.toLowerCase().includes(query.toLowerCase()));
};

export const fetchData = async (symbol: string): Promise<DataResponse> => {
  await new Promise(resolve => setTimeout(resolve, 500));

  // 1. FETCH: Simulate SELECT * FROM currency WHERE pair = symbol
  const currencyRows = generateCurrencyTableData(100, symbol);

  // 2. MAP: Convert DB rows to Frontend Candle format
  const candles: Candle[] = currencyRows.map(row => ({
      time: row.day,
      open: row.open,
      high: row.high,
      low: row.low,
      close: row.close
  }));

  // 3. FETCH: Trades (from orders table)
  const trades = getTradesForSymbol(symbol, currencyRows);
  
  // 4. CALCULATE: Pivots (or fetch if stored in another table)
  const currentPrice = candles[candles.length - 1].close;
  const pivots = generatePivots(currentPrice, symbol);

  // Optional: You could also return EMA data here since your DB has it
  const indicators = {
      ema_fast: currencyRows.map(r => ({ time: r.day, value: r.ema_fast })),
      ema_slow: currencyRows.map(r => ({ time: r.day, value: r.ema_slow })),
  };

  return {
    candles,
    trades,
    pivots,
    indicators
  };
};

export const fetchPortfolio = async (): Promise<Trade[]> => {
    await new Promise(resolve => setTimeout(resolve, 400));
    // Simulates: SELECT * FROM orders WHERE status IN ('OPEN', 'PENDING')
    return MOCK_ORDERS_TABLE.filter(t => t.status === 'OPEN' || t.status === 'PENDING');
};
