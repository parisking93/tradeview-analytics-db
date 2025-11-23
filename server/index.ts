import express from "express";
import cors from "cors";
import { pool } from "./db";

const app = express();
app.use(cors());
app.use(express.json());

const mapTimeframe = (tf?: string) => {
  if (!tf) return "5"; // default
  const t = tf.toLowerCase();
  const m: Record<string,string> = {
    "1m": "1",
    "5m": "5",
    "15m": "15",
    "30m": "30",
    "1h": "60",
    "24h": "D",
    "1d": "D",
  };
  return m[t] || tf; // se già "1","5","D", ecc.
};

app.get("/api/health", (_, res) => res.json({ ok: true }));

// Cerca simboli nel DB (per la tua searchbar)
app.get("/api/pairs", async (req, res) => {
  const query = String(req.query.query || "").trim();
  if (!query) return res.json([]);

  const like = `%${query}%`;
  const [rows] = await pool.query(
    `SELECT DISTINCT symbol 
     FROM market_data 
     WHERE symbol LIKE ?
     ORDER BY symbol
     LIMIT 50`,
    [like]
  );
  res.json((rows as any[]).map(r => r.symbol));
});

// Ritorna candles dal DB
app.get("/api/market/:symbol", async (req, res) => {
  const symbol = String(req.params.symbol);
  const resolution = mapTimeframe(String(req.query.timeframe || req.query.resolution || "5"));
  const limit = Math.min(Number(req.query.limit || 500), 2000);

  // fallback: se in frontend mandi "BTCUSD" ma nel DB c'è "BINANCE:BTCUSDT"
  const like = `%${symbol}%`;

  const [rows] = await pool.query(
    `SELECT 
        ts, dt, open, high, low, close, volume, ema, bid, ask, last, spread,
        resolution, asset_type
     FROM market_data
     WHERE (symbol = ? OR symbol LIKE ?)
       AND resolution = ?
     ORDER BY ts DESC
     LIMIT ?`,
    [symbol, like, resolution, limit]
  );

  const data = (rows as any[]).reverse(); // ordine cronologico per chart
  res.json(data);
});

// Placeholder portfolio/trades: per ora vuoto finché non hai tabella ordini
app.get("/api/portfolio", async (_req, res) => res.json([]));

const PORT = Number(process.env.API_PORT || 3001);
app.listen(PORT, () => console.log(`API listening on http://localhost:${PORT}`));
