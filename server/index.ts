// --- START OF FILE server/index.ts ---

import cors from "cors";
import express from "express";
import { pool } from "./db";

const app = express();
app.use(cors());
app.use(express.json());

const normalizeTimeframe = (tf?: string) => {
  if (!tf) return "5m";
  const t = tf.toLowerCase();
  const map: Record<string, string> = {
    "1": "1m", "1m": "1m", "5": "5m", "5m": "5m",
    "15": "15m", "15m": "15m", "30": "30m", "30m": "30m",
    "60": "1h", "1h": "1h", "240": "4h", "4h": "4h",
    "d": "1d", "1d": "1d", "24h": "1d",
  };
  return map[t] || t;
};

const forecastTimeframe = (tf?: string) => {
  const base = normalizeTimeframe(tf);
  return base.includes('+1') ? base : `${base}+1`;
};

const compactSymbol = (value: string) =>
  value.replace(/[^A-Za-z0-9]/g, "").toUpperCase();

const resolvePairFromLimits = async (symbol: string) => {
  const compact = compactSymbol(symbol);
  const [rows] = await pool.query(
    `SELECT pair, kr_pair, base, quote
     FROM pair_limits
     WHERE pair = ? OR kr_pair = ? OR REPLACE(pair, '/', '') = ? OR CONCAT(base, quote) = ? OR pair LIKE ? OR kr_pair LIKE ?
     ORDER BY pair LIMIT 1`,
    [symbol, symbol, compact, compact, `%${symbol}%`, `%${symbol}%`]
  );
  if (!(rows as any[]).length) return { pair: symbol, kr_pair: symbol, base: "", quote: "" };
  const row = (rows as any[])[0];
  return { pair: row.pair || symbol, kr_pair: row.kr_pair || symbol, base: row.base || "", quote: row.quote || "" };
};

const mapOrderRowToTrade = (row: any) => {
  // ... (tua logica esistente di mapping)
  const symbol = row.pair || row.kr_pair || `${row.base || ""}${row.quote || ""}`.replace(/[^A-Za-z0-9/:-]/g, "");
  const rawType = String(row.type || "").toLowerCase();
  const type = rawType.includes("sell") || rawType.includes("short") ? "SHORT" : "LONG";
  const rawStatus = String(row.subtype || "").toUpperCase();
  const status = rawStatus === "CLOSED" ? "CLOSED" : rawStatus === "PENDING" ? "PENDING" : "OPEN";
  const dateSource = row.record_date || row.created_at;
  const entryDate = dateSource ? new Date(dateSource).toISOString().slice(0, 10) : "";
  const entryPrice = row.price_entry ?? row.price_avg ?? row.price ?? row.value_eur ?? 0;
  const createdAt = row.created_at ? new Date(row.created_at).toISOString() : undefined;
  return {
    id: Number(row.id),
    symbol, entryDate, type, entryPrice: Number(entryPrice),
    stopLoss: Number(row.stop_loss || 0), takeProfit: Number(row.take_profit || 0),
    status, pnl: row.pnl !== null && row.pnl !== undefined ? Number(row.pnl) : undefined,
    createdAt
  };
};

app.get("/api/health", (_req, res) => res.json({ ok: true }));

app.get("/api/pairs", async (req, res) => {
  // ... (tua logica search esistente)
  const query = String(req.query.query || "").trim();
  if (!query) return res.json([]);
  const like = `%${query}%`;
  try {
    const [limitsRows] = await pool.query(
      `SELECT DISTINCT pair, base, quote FROM pair_limits WHERE pair LIKE ? OR base LIKE ? OR quote LIKE ? ORDER BY pair LIMIT 50`,
      [like, like, like]
    );
    const pairsFromLimits = (limitsRows as any[]).map((r) => r.pair || `${r.base || ""}${r.quote || ""}`).filter(Boolean);
    const [currencyRows] = await pool.query(
      `SELECT DISTINCT pair FROM currency WHERE pair LIKE ? OR kr_pair LIKE ? ORDER BY pair LIMIT 50`,
      [like, like]
    );
    const pairsFromCurrency = (currencyRows as any[]).map((r) => r.pair).filter(Boolean);
    const merged = Array.from(new Set([...pairsFromLimits, ...pairsFromCurrency]));
    return res.json(merged);
  } catch (error) {
    return res.status(500).json([]);
  }
});

// --- API MARKET AGGIORNATA ---
app.get("/api/market/:symbol", async (req, res) => {
  const symbol = String(req.params.symbol).trim();
  const timeframe = normalizeTimeframe(String(req.query.timeframe || req.query.resolution || "5m"));
  const toDate = req.query.to ? String(req.query.to) : null;
  const limit = Math.min(Number(req.query.limit || 800), 5000);
  const compact = compactSymbol(symbol);

  try {
    const resolved = await resolvePairFromLimits(symbol);
    const targetTimeframe = req.query.timeframe ? timeframe : timeframe;
    const targetForecastTimeframe = forecastTimeframe(targetTimeframe);

    // Query Base
    let querySQL = `
      SELECT
          pair, kr_pair, base, quote, timestamp, open, high, low, close, volume,
          ema_fast, ema_slow, rsi, atr, bid, ask, mid, spread, timeframe
       FROM currency
       WHERE (pair = ? OR kr_pair = ? OR REPLACE(pair, '/', '') = ? OR pair LIKE ?)
         AND timeframe = ?
    `;
    const queryParams: any[] = [resolved.pair, resolved.kr_pair, compact, `%${symbol}%`, targetTimeframe];

    // Logica Load More (Infinito Scroll)
    if (toDate) {
      // Importante: usiamo STR_TO_DATE per assicurarci che il confronto sia su Date e non Stringhe
      querySQL += ` AND STR_TO_DATE(timestamp, '%Y-%m-%d %H:%i:%s') < STR_TO_DATE(?, '%Y-%m-%d %H:%i:%s') `;
      queryParams.push(toDate);
    }

    // Ordinamento DESC per prendere i più vicini alla data (o a NOW), poi LIMIT
    querySQL += ` ORDER BY STR_TO_DATE(timestamp, '%Y-%m-%d %H:%i:%s') DESC LIMIT ?`;
    queryParams.push(limit);

    const [rows] = await pool.query(querySQL, queryParams);

    // --- Forecast query (timeframe +1) ---
    let forecastSQL = `
      SELECT
          pair, kr_pair, base, quote, timestamp, open, high, low, close, volume,
          ema_fast, ema_slow, rsi, atr, bid, ask, mid, spread, timeframe
       FROM forecast
       WHERE (pair = ? OR kr_pair = ? OR REPLACE(pair, '/', '') = ? OR pair LIKE ?)
         AND timeframe = ?
    `;
    const forecastParams: any[] = [resolved.pair, resolved.kr_pair, compact, `%${symbol}%`, targetForecastTimeframe];

    if (toDate) {
      forecastSQL += ` AND STR_TO_DATE(timestamp, '%Y-%m-%d %H:%i:%s') < STR_TO_DATE(?, '%Y-%m-%d %H:%i:%s') `;
      forecastParams.push(toDate);
    }

    forecastSQL += ` ORDER BY STR_TO_DATE(timestamp, '%Y-%m-%d %H:%i:%s') DESC LIMIT ?`;
    forecastParams.push(limit);

    let [forecastRows] = await pool.query(forecastSQL, forecastParams);

    // Fallback: se non ci sono previsioni per quel timeframe, prende le più recenti per la coppia
    if (!(forecastRows as any[]).length) {
      const fallbackForecastSQL = `
        SELECT pair, kr_pair, base, quote, timestamp, open, high, low, close, volume,
               ema_fast, ema_slow, rsi, atr, bid, ask, mid, spread, timeframe
        FROM forecast
        WHERE (pair = ? OR kr_pair = ? OR REPLACE(pair, '/', '') = ? OR pair LIKE ?)
        ORDER BY STR_TO_DATE(timestamp, '%Y-%m-%d %H:%i:%s') DESC
        LIMIT ?
      `;
      forecastRows = (await pool.query(fallbackForecastSQL, [resolved.pair, resolved.kr_pair, compact, `%${symbol}%`, limit]))[0] as any[];
    }

    if (!(rows as any[]).length && !(forecastRows as any[]).length) {
      return res.json({ symbol: resolved.pair || symbol, timeframe: targetTimeframe, candles: [], forecast: [], indicators: {}, lastPrice: 0, count: 0 });
    }

    const rowsAsList = (rows as any[]).map((row) => ({
      pair: row.pair,
      timestamp: row.timestamp || row.dt || row.ts,
      open: Number(row.open),
      high: Number(row.high),
      low: Number(row.low),
      close: Number(row.close),
      // ... indicatori
      ema_fast: row.ema_fast ? Number(row.ema_fast) : undefined,
      ema_slow: row.ema_slow ? Number(row.ema_slow) : undefined,
    }));

    const forecastRowsAsList = (forecastRows as any[]).map((row) => ({
      pair: row.pair,
      timestamp: row.timestamp || row.dt || row.ts,
      open: Number(row.open),
      high: Number(row.high),
      low: Number(row.low),
      close: Number(row.close),
    }));

    // Invertiamo l'array (dal DB arriva: Oggi, Ieri, L'altro ieri...)
    // Il frontend vuole: (L'altro ieri, Ieri, Oggi...)
    const ordered = rowsAsList.reverse();
    const forecastOrdered = forecastRowsAsList.reverse();

    const candles = ordered.map((row) => ({
      time: row.timestamp,
      open: row.open,
      high: row.high,
      low: row.low,
      close: row.close,
    }));

    const indicators = {
      ema_fast: ordered.filter(r => r.ema_fast).map(r => ({ time: r.timestamp, value: r.ema_fast })),
      ema_slow: ordered.filter(r => r.ema_slow).map(r => ({ time: r.timestamp, value: r.ema_slow }))
    };

    const forecastCandles = forecastOrdered.map((row) => ({
      time: row.timestamp,
      open: row.open,
      high: row.high,
      low: row.low,
      close: row.close,
    }));

    return res.json({
      symbol: resolved.pair || symbol,
      timeframe: targetTimeframe,
      candles,
      forecast: forecastCandles,
      indicators,
      lastPrice: candles.length ? candles[candles.length - 1].close : 0,
      count: candles.length,
    });

  } catch (error) {
    console.error("[/api/market] error", error);
    return res.status(500).json({ error: "Db Error" });
  }
});

// ... Portfolio endpoint
app.get("/api/portfolio", async (_req, res) => {
  // ... invariato
  const [rows] = await pool.query(`SELECT * FROM orders WHERE created_at > NOW() - INTERVAL 2 DAY ORDER BY created_at DESC`);
  const trades = (rows as any[]).map(mapOrderRowToTrade);
  return res.json(trades);
});

const PORT = Number(process.env.API_PORT || 3001);
app.listen(PORT, () => console.log(`API listening on port ${PORT}`));
