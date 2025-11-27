import express from "express";
import cors from "cors";
import { pool } from "./db";

const app = express();
app.use(cors());
app.use(express.json());

const normalizeTimeframe = (tf?: string) => {
  if (!tf) return "5m";
  const t = tf.toLowerCase();
  const map: Record<string, string> = {
    "1": "1m",
    "1m": "1m",
    "5": "5m",
    "5m": "5m",
    "15": "15m",
    "15m": "15m",
    "30": "30m",
    "30m": "30m",
    "60": "1h",
    "1h": "1h",
    "240": "4h",
    "4h": "4h",
    "d": "1d",
    "1d": "1d",
    "24h": "1d",
  };
  return map[t] || t;
};

const compactSymbol = (value: string) =>
  value.replace(/[^A-Za-z0-9]/g, "").toUpperCase();

const resolvePairFromLimits = async (symbol: string) => {
  const compact = compactSymbol(symbol);

  const [rows] = await pool.query(
    `SELECT pair, kr_pair, base, quote
     FROM pair_limits
     WHERE pair = ? 
        OR kr_pair = ?
        OR REPLACE(pair, '/', '') = ?
        OR CONCAT(base, quote) = ?
        OR pair LIKE ?
        OR kr_pair LIKE ?
     ORDER BY pair LIMIT 1`,
    [symbol, symbol, compact, compact, `%${symbol}%`, `%${symbol}%`]
  );

  if (!(rows as any[]).length) {
    return { pair: symbol, kr_pair: symbol, base: "", quote: "" };
  }

  const row = (rows as any[])[0];
  return {
    pair: row.pair || symbol,
    kr_pair: row.kr_pair || symbol,
    base: row.base || "",
    quote: row.quote || "",
  };
};

const mapOrderRowToTrade = (row: any) => {
  const symbol =
    row.pair ||
    row.kr_pair ||
    `${row.base || ""}${row.quote || ""}`.replace(/[^A-Za-z0-9/:-]/g, "");

  const rawType = String(row.type || "").toLowerCase();
  const type =
    rawType.includes("sell") || rawType.includes("short") ? "SHORT" : "LONG";

  const rawStatus = String(row.subtype || "").toUpperCase();
  const status =
    rawStatus === "CLOSED"
      ? "CLOSED"
      : rawStatus === "PENDING"
      ? "PENDING"
      : "OPEN";

  const dateSource = row.record_date || row.created_at;
  const entryDate = dateSource
    ? new Date(dateSource).toISOString().slice(0, 10)
    : "";

  const entryPrice =
    row.price_entry ?? row.price_avg ?? row.price ?? row.value_eur ?? 0;

  const takeProfit =
    row.take_profit !== null && row.take_profit !== undefined
      ? Number(row.take_profit)
      : 0;
  const stopLoss =
    row.stop_loss !== null && row.stop_loss !== undefined
      ? Number(row.stop_loss)
      : 0;

  return {
    id: Number(row.id),
    symbol,
    entryDate,
    type,
    entryPrice: Number(entryPrice),
    stopLoss,
    takeProfit,
    status,
    pnl: row.pnl !== null && row.pnl !== undefined ? Number(row.pnl) : undefined,
  };
};

app.get("/api/health", (_req, res) => res.json({ ok: true }));

// Cerca simboli nel DB (per la searchbar del FE)
app.get("/api/pairs", async (req, res) => {
  const query = String(req.query.query || "").trim();
  if (!query) return res.json([]);

  const like = `%${query}%`;

  try {
    const [limitsRows] = await pool.query(
      `SELECT DISTINCT pair, base, quote
       FROM pair_limits
       WHERE pair LIKE ? OR base LIKE ? OR quote LIKE ?
       ORDER BY pair
       LIMIT 50`,
      [like, like, like]
    );

    const pairsFromLimits = (limitsRows as any[])
      .map((r) => r.pair || `${r.base || ""}${r.quote || ""}`)
      .filter(Boolean);

    const [currencyRows] = await pool.query(
      `SELECT DISTINCT pair 
       FROM currency 
       WHERE pair LIKE ? OR kr_pair LIKE ?
       ORDER BY pair
       LIMIT 50`,
      [like, like]
    );

    const pairsFromCurrency = (currencyRows as any[])
      .map((r) => r.pair)
      .filter(Boolean);

    const merged = Array.from(new Set([...pairsFromLimits, ...pairsFromCurrency]));
    return res.json(merged);
  } catch (error) {
    console.error("[/api/pairs] error", error);
    return res.status(500).json([]);
  }
});

// Restituisce le candele della tabella currency
app.get("/api/market/:symbol", async (req, res) => {
  const symbol = String(req.params.symbol).trim();
  const timeframe = normalizeTimeframe(
    String(req.query.timeframe || req.query.resolution || "5m")
  );
  const limit = Math.min(Number(req.query.limit || 800), 2000);
  const compact = compactSymbol(symbol);

  try {
    const resolved = await resolvePairFromLimits(symbol);

    const targetTimeframe =
      req.query.timeframe || req.query.resolution
        ? timeframe
        : await (async () => {
            const [tfRows] = await pool.query(
              `SELECT timeframe
               FROM currency
               WHERE (pair = ? OR kr_pair = ? OR REPLACE(pair, '/', '') = ? OR pair LIKE ?)
               ORDER BY created_at DESC
               LIMIT 1`,
              [resolved.pair, resolved.kr_pair, compact, `%${symbol}%`]
            );
            if ((tfRows as any[]).length && (tfRows as any[])[0].timeframe) {
              return normalizeTimeframe((tfRows as any[])[0].timeframe);
            }
            return timeframe;
          })();

    const [rows] = await pool.query(
      `SELECT 
          pair, kr_pair, base, quote, timestamp, open, high, low, close, volume,
          ema_fast, ema_slow, rsi, atr, bid, ask, mid, spread, timeframe
       FROM currency
       WHERE (pair = ? OR kr_pair = ? OR REPLACE(pair, '/', '') = ? OR pair LIKE ?)
         AND timeframe = ?
       ORDER BY timestamp DESC
       LIMIT ?`,
      [resolved.pair, resolved.kr_pair, compact, `%${symbol}%`, targetTimeframe, limit]
    );

    if (!(rows as any[]).length) {
      return res.json({
        symbol: resolved.pair || symbol,
        timeframe: targetTimeframe,
        candles: [],
        indicators: { ema_fast: [], ema_slow: [] },
        lastPrice: 0,
        spread: undefined,
        count: 0,
      });
    }

    const rowsAsList = (rows as any[]).map((row) => ({
      pair: row.pair,
      kr_pair: row.kr_pair,
      base: row.base,
      quote: row.quote,
      timestamp: row.timestamp || row.dt || row.ts,
      open: Number(row.open),
      high: Number(row.high),
      low: Number(row.low),
      close: Number(row.close),
      volume:
        row.volume !== null && row.volume !== undefined ? Number(row.volume) : 0,
      ema_fast:
        row.ema_fast !== null && row.ema_fast !== undefined
          ? Number(row.ema_fast)
          : undefined,
      ema_slow:
        row.ema_slow !== null && row.ema_slow !== undefined
          ? Number(row.ema_slow)
          : undefined,
      rsi:
        row.rsi !== null && row.rsi !== undefined ? Number(row.rsi) : undefined,
      atr:
        row.atr !== null && row.atr !== undefined ? Number(row.atr) : undefined,
      bid: row.bid !== null && row.bid !== undefined ? Number(row.bid) : undefined,
      ask: row.ask !== null && row.ask !== undefined ? Number(row.ask) : undefined,
      mid: row.mid !== null && row.mid !== undefined ? Number(row.mid) : undefined,
      spread:
        row.spread !== null && row.spread !== undefined
          ? Number(row.spread)
          : undefined,
      timeframe: row.timeframe || targetTimeframe,
    }));

    const ordered = rowsAsList.reverse();

    const candles = ordered.map((row) => ({
      time: row.timestamp,
      open: row.open,
      high: row.high,
      low: row.low,
      close: row.close,
    }));

    const indicators = {
      ema_fast: ordered
        .filter((row) => row.ema_fast !== undefined)
        .map((row) => ({ time: row.timestamp, value: row.ema_fast as number })),
      ema_slow: ordered
        .filter((row) => row.ema_slow !== undefined)
        .map((row) => ({ time: row.timestamp, value: row.ema_slow as number })),
    };

    const lastClose = candles.length ? candles[candles.length - 1].close : 0;
    const spreadSample = ordered.length
      ? ordered[ordered.length - 1].spread
      : undefined;

    return res.json({
      symbol: resolved.pair || symbol,
      timeframe: targetTimeframe,
      candles,
      indicators,
      lastPrice: lastClose,
      spread: spreadSample,
      count: candles.length,
    });
  } catch (error) {
    console.error("[/api/market/:symbol] error", error);
    return res.status(500).json({ error: "Failed to load market data" });
  }
});

// Portfolio / Orders
app.get("/api/portfolio", async (_req, res) => {
  try {
    const [rows] = await pool.query(
      `SELECT 
         id, pair, kr_pair, base, quote, qty, price_entry, price_avg, take_profit,
         stop_loss, price, value_eur, pnl, type, subtype, created_at, record_date
       FROM orders
       ORDER BY created_at DESC`
    );

    const trades = (rows as any[]).map(mapOrderRowToTrade);
    return res.json(trades);
  } catch (error) {
    console.error("[/api/portfolio] error", error);
    return res.status(500).json([]);
  }
});

const PORT = Number(process.env.API_PORT || 3001);
app.listen(PORT, () => console.log(`API listening on http://localhost:${PORT}`));
