import time
import logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional, Literal, Dict, Any, List
from dotenv import load_dotenv
import os
import requests
import pandas as pd
import mysql.connector
from mysql.connector import Error as MySQLError


AssetType = Literal["stock", "crypto", "forex"]

@dataclass
class MySQLConfig:
    host: str = "127.0.0.1"
    user: str = "root"
    password: str = ""
    database: str = "trading"
    port: int = 3306


class FinnhubMySQLIngestor:
    BASE_URL = "https://finnhub.io/api/v1"

    def __init__(
        self,
        api_key: str,
        mysql_cfg: MySQLConfig,
        table: str = "market_data",
        max_retries: int = 5,
        backoff_sec: float = 0.8,
        timeout_sec: int = 20,
        logger: Optional[logging.Logger] = None,
    ):
        self.api_key = api_key
        self.mysql_cfg = mysql_cfg
        self.table = table
        self.max_retries = max_retries
        self.backoff_sec = backoff_sec
        self.timeout_sec = timeout_sec
        self.sess = requests.Session()
        self.log = logger or logging.getLogger("FinnhubMySQLIngestor")
        if not self.log.handlers:
            logging.basicConfig(level=logging.INFO)

    # ---------- HTTP helpers ----------
    def _request(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.BASE_URL}{path}"
        params = dict(params)
        params["token"] = self.api_key

        for attempt in range(1, self.max_retries + 1):
            try:
                r = self.sess.get(url, params=params, timeout=self.timeout_sec)
                if r.status_code == 429:
                    sleep = self.backoff_sec * attempt
                    self.log.warning(f"429 rate limit. sleep {sleep:.1f}s")
                    time.sleep(sleep)
                    continue
                r.raise_for_status()
                return r.json()
            except Exception as e:
                if attempt == self.max_retries:
                    raise
                sleep = self.backoff_sec * attempt
                self.log.warning(f"HTTP error {e}. retry in {sleep:.1f}s")
                time.sleep(sleep)

        return {}

    # ---------- Finnhub fetchers ----------
    def fetch_candles(
        self,
        symbol: str,
        asset_type: AssetType,
        resolution: str,
        lookback_days: int = 5,
    ) -> pd.DataFrame:
        """
        Returns DataFrame with columns:
        ts, dt, open, high, low, close, volume
        """
        now = datetime.now(timezone.utc)
        frm = int((now - timedelta(days=lookback_days)).timestamp())
        to = int(now.timestamp())

        endpoint = {
            "stock": "/stock/candle",
            "crypto": "/crypto/candle",
            "forex": "/forex/candle",
        }[asset_type]

        data = self._request(endpoint, {
            "symbol": symbol,
            "resolution": resolution,
            "from": frm,
            "to": to
        })

        if not data or data.get("s") != "ok":
            self.log.warning(f"No candles for {symbol} ({asset_type}) res={resolution}: {data}")
            return pd.DataFrame()

        df = pd.DataFrame({
            "ts": data["t"],
            "open": data["o"],
            "high": data["h"],
            "low": data["l"],
            "close": data["c"],
            "volume": data.get("v", [None]*len(data["t"]))
        })
        df["dt"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert(None)
        return df

    def fetch_quote(self, symbol: str) -> Dict[str, Any]:
        """
        /quote works for stock/forex/crypto.
        Returns current / last price etc.
        """
        return self._request("/quote", {"symbol": symbol})

    def fetch_bidask_stock(self, symbol: str) -> Dict[str, Any]:
        """
        Premium for US stocks. On free may fail/return empty.
        """
        try:
            return self._request("/stock/bidask", {"symbol": symbol})
        except Exception as e:
            self.log.info(f"bidask not available for {symbol}: {e}")
            return {}

    # ---------- Indicators ----------
    @staticmethod
    def compute_ema(close_series: pd.Series, period: int = 50) -> pd.Series:
        return close_series.ewm(span=period, adjust=False).mean()

    # ---------- MySQL ----------
    def _connect_mysql(self):
        return mysql.connector.connect(
            host=self.mysql_cfg.host,
            user=self.mysql_cfg.user,
            password=self.mysql_cfg.password,
            database=self.mysql_cfg.database,
            port=self.mysql_cfg.port,
            autocommit=False,
        )

    def upsert_dataframe(
        self,
        df: pd.DataFrame,
        symbol: str,
        asset_type: AssetType,
        resolution: str,
        bid: Optional[float],
        ask: Optional[float],
        last: Optional[float],
        spread: Optional[float],
    ):
        if df.empty:
            return

        insert_sql = f"""
        INSERT INTO {self.table} (
            symbol, asset_type, resolution, ts, dt,
            open, high, low, close, volume,
            bid, ask, last, spread, ema
        ) VALUES (
            %s,%s,%s,%s,%s,
            %s,%s,%s,%s,%s,
            %s,%s,%s,%s,%s
        )
        ON DUPLICATE KEY UPDATE
            open=VALUES(open),
            high=VALUES(high),
            low=VALUES(low),
            close=VALUES(close),
            volume=VALUES(volume),
            bid=VALUES(bid),
            ask=VALUES(ask),
            last=VALUES(last),
            spread=VALUES(spread),
            ema=VALUES(ema)
        """
        rows = []
        for _, r in df.iterrows():
            rows.append((
                symbol, asset_type, resolution, int(r.ts), r.dt,
                float(r.open) if pd.notna(r.open) else None,
                float(r.high) if pd.notna(r.high) else None,
                float(r.low) if pd.notna(r.low) else None,
                float(r.close) if pd.notna(r.close) else None,
                float(r.volume) if pd.notna(r.volume) else None,
                bid, ask, last, spread,
                float(r.ema) if pd.notna(r.ema) else None
            ))

        conn = None
        try:
            conn = self._connect_mysql()
            cur = conn.cursor()
            cur.executemany(insert_sql, rows)
            conn.commit()
            self.log.info(f"Upsert ok: {symbol} {asset_type} res={resolution} rows={len(rows)}")
        except MySQLError as e:
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()

    # ---------- High-level pipeline ----------
    def fetch_and_store(
        self,
        symbol: str,
        asset_type: AssetType,
        timeframe: str,
        lookback_days: int = 5,
        ema_period: int = 50,
    ):
        """
        timeframe examples: '1m', '5m', '15m', '30m', '1h', '24h'
        """
        resolution = self._map_timeframe(timeframe)

        # 1) candles
        df = self.fetch_candles(symbol, asset_type, resolution, lookback_days=lookback_days)
        if df.empty:
            return

        # 2) EMA
        df["ema"] = self.compute_ema(df["close"], ema_period)

        # 3) quote/last
        q = self.fetch_quote(symbol)
        last = q.get("c")

        # 4) bid/ask (solo stock, premium). Per crypto/forex lascia None oppure estendi tu.
        bid = ask = None
        if asset_type == "stock":
            ba = self.fetch_bidask_stock(symbol)
            # alcuni account tornano liste, altri valore singolo. gestiamo entrambi.
            if isinstance(ba.get("b"), list) and ba["b"]:
                bid = ba["b"][0]
            elif ba.get("b") is not None:
                bid = ba["b"]

            if isinstance(ba.get("a"), list) and ba["a"]:
                ask = ba["a"][0]
            elif ba.get("a") is not None:
                ask = ba["a"]

        spread = (ask - bid) if (ask is not None and bid is not None) else None

        # 5) upsert
        self.upsert_dataframe(df, symbol, asset_type, resolution, bid, ask, last, spread)

    @staticmethod
    def _map_timeframe(tf: str) -> str:
        tf = tf.strip().lower()
        mapping = {
            "1m": "1",
            "5m": "5",
            "15m": "15",
            "30m": "30",
            "1h": "60",
            "60m": "60",
            "24h": "D",
            "1d": "D",
            "d": "D",
            "w": "W",
            "m": "M",
        }
        if tf not in mapping:
            raise ValueError(f"Unsupported timeframe '{tf}'. Use one of {list(mapping.keys())}")
        return mapping[tf]

    def ingest_many(
        self,
        symbols: List[str],
        asset_type: AssetType,
        timeframe: str,
        lookback_days: int = 5,
        ema_period: int = 50,
        sleep_sec: float = 1.0,  # per rispettare rate limit free
    ):
        for s in symbols:
            self.fetch_and_store(
                symbol=s,
                asset_type=asset_type,
                timeframe=timeframe,
                lookback_days=lookback_days,
                ema_period=ema_period,
            )
            time.sleep(sleep_sec)

if __name__ == "__main__":

    API_KEY = os.getenv("FINNHUB_API_KEY")
    load_dotenv(".env.python")
    mysql_cfg = MySQLConfig(
        host=os.getenv("MYSQL_HOST","127.0.0.1"),
        user=os.getenv("MYSQL_USER","root"),
        password=os.getenv("MYSQL_PASSWORD",""),
        database=os.getenv("MYSQL_DATABASE","trading"),
        port=int(os.getenv("MYSQL_PORT","3306")),
    )

    ing = FinnhubMySQLIngestor(API_KEY, mysql_cfg)
    # AZIONI (simbolo semplice)
    ing.fetch_and_store("AAPL", asset_type="stock", timeframe="5m", lookback_days=10, ema_period=50)
    # CRYPTO (simbolo Finnhub con exchange)
    ing.fetch_and_store("BINANCE:BTCUSDT", asset_type="crypto", timeframe="15m", lookback_days=5)
    # FOREX (simbolo Finnhub forex)
    ing.fetch_and_store("OANDA:EUR_USD", asset_type="forex", timeframe="1h", lookback_days=30)

    # while True:
    #     ing.fetch_and_store("AAPL", asset_type="stock", timeframe="5m", lookback_days=10, ema_period=50)
    #     ing.fetch_and_store("BINANCE:BTCUSDT", asset_type="crypto", timeframe="15m", lookback_days=5)
    #     ing.fetch_and_store("OANDA:EUR_USD", asset_type="forex", timeframe="1h", lookback_days=30)
    #     time.sleep(60)  # ogni minuto

# CREATE TABLE IF NOT EXISTS market_data (
#   id BIGINT AUTO_INCREMENT PRIMARY KEY,
#   symbol VARCHAR(64) NOT NULL,
#   asset_type ENUM('stock','crypto','forex') NOT NULL,
#   resolution VARCHAR(8) NOT NULL,
#   ts INT NOT NULL,                -- unix seconds
#   dt DATETIME NOT NULL,           -- UTC datetime

#   open DOUBLE NULL,
#   high DOUBLE NULL,
#   low DOUBLE NULL,
#   close DOUBLE NULL,
#   volume DOUBLE NULL,

#   bid DOUBLE NULL,
#   ask DOUBLE NULL,
#   last DOUBLE NULL,
#   spread DOUBLE NULL,
#   ema DOUBLE NULL,

#   ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
#   UNIQUE KEY uniq_symbol_res_ts (symbol, asset_type, resolution, ts)
# ) ENGINE=InnoDB;
