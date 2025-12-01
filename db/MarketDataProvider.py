# --- START OF FILE MarketDataProvider.py ---

import asyncio
import json
import os
import time
import datetime
import pandas as pd
import numpy as np  # Necessario per i calcoli vettoriali
import yfinance as yf
import krakenex
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Tentiamo l'import, ma non rendiamo il modulo inutilizzabile se websockets non è installato.
try:
    import websockets  # type: ignore
except ImportError:  # pragma: no cover - dipendenza opzionale
    websockets = None

class MarketDataProvider:
    def __init__(self, kraken_map: Dict[str, str] = None, yahoo_map: Dict[str, str] = None):
        """Inizializza il provider dati."""
        load_dotenv()

        kraken_key = os.getenv("KRAKEN_KEY")
        kraken_secret = os.getenv("KRAKEN_SECRET")
        self.k = krakenex.API(key=kraken_key, secret=kraken_secret)

        self.yahoo_map = yahoo_map if yahoo_map else {}
        self.kraken_pairs_map = kraken_map if kraken_map else {}
        self.pair_metadata = {}

        if not self.kraken_pairs_map:
            self._load_kraken_asset_pairs()

        # Parametri indicatori
        self.ema_fast_span = 12
        self.ema_slow_span = 26
        self.rsi_period = 14
        self.atr_period = 14

    def _load_kraken_asset_pairs(self):
        """Scarica le coppie da Kraken e costruisce la mappa di risoluzione e metadati."""
        try:
            response = self.k.query_public('AssetPairs')
            if response.get('error'): return

            results = response['result']
            for pair_id, info in results.items():
                wsname = info.get('wsname')
                altname = info.get('altname')

                self.kraken_pairs_map[pair_id] = pair_id
                if wsname:
                    self.kraken_pairs_map[wsname] = pair_id
                    if 'XBT' in wsname:
                        btc_alias = wsname.replace('XBT', 'BTC')
                        self.kraken_pairs_map[btc_alias] = pair_id

                if altname:
                    self.kraken_pairs_map[altname] = pair_id

                self.pair_metadata[pair_id] = {
                    'base': info.get('base'),
                    'quote': info.get('quote'),
                    'wsname': wsname,
                    'altname': altname,
                    'lot_decimals': info.get('lot_decimals'),
                    'pair_decimals': info.get('pair_decimals'),
                    'ordermin': info.get('ordermin'),
                    'leverage_buy': info.get('leverage_buy', []),
                    'leverage_sell': info.get('leverage_sell', []),
                    'fees': info.get('fees', []),
                    'fees_maker': info.get('fees_maker', []),
                    'fee_volume_currency': info.get('fee_volume_currency')
                }
        except Exception:
            pass

    def _get_kraken_id(self, pair: str) -> str:
        if pair in self.kraken_pairs_map: return self.kraken_pairs_map[pair]
        if "BTC" in pair:
            xbt_variant = pair.replace("BTC", "XBT")
            if xbt_variant in self.kraken_pairs_map:
                return self.kraken_pairs_map[xbt_variant]
        return pair

    def getPair(self, pair: str) -> Dict[str, Any]:
        k_id = self._get_kraken_id(pair)
        meta = self.pair_metadata.get(k_id, {})
        if not meta:
            return {"base": "N/A", "quote": "N/A", "pair": pair, "kr_pair": k_id, "pair_limits": None}

        base_clean = "N/A"
        quote_clean = "N/A"
        human_pair = meta.get('wsname', pair)

        if human_pair and '/' in human_pair:
            base_clean, quote_clean = human_pair.split('/')
            if base_clean == 'XBT': base_clean = 'BTC'
        else:
            raw_base = meta.get('base', '')
            raw_quote = meta.get('quote', '')
            base_clean = raw_base
            if len(raw_base) == 4 and raw_base.startswith('X'): base_clean = raw_base[1:]
            if raw_base in ['XXBT', 'XBT']: base_clean = 'BTC'
            quote_clean = raw_quote
            if len(raw_quote) == 4 and raw_quote.startswith('Z'): quote_clean = raw_quote[1:]
            if raw_quote in ['ZEUR']: quote_clean = 'EUR'
            if raw_quote in ['ZUSD']: quote_clean = 'USD'
            if base_clean and quote_clean: human_pair = f"{base_clean}/{quote_clean}"

        lev_buy = meta.get('leverage_buy', [])
        lev_sell = meta.get('leverage_sell', [])

        pair_limits = {
            "lot_decimals": meta.get('lot_decimals'),
            "pair_decimals": meta.get('pair_decimals'),
            "ordermin": meta.get('ordermin'),
            "fee_volume_currency": meta.get('fee_volume_currency'),
            "fees": meta.get('fees', []),
            "fees_maker": meta.get('fees_maker', []),
            "leverage_buy": lev_buy,
            "leverage_sell": lev_sell,
            "leverage_buy_max": max(lev_buy) if lev_buy else 0,
            "leverage_sell_max": max(lev_sell) if lev_sell else 0,
            "can_leverage_buy": bool(lev_buy),
            "can_leverage_sell": bool(lev_sell)
        }

        return {"base": base_clean, "quote": quote_clean, "pair": human_pair, "kr_pair": k_id, "pair_limits": pair_limits}

    def _get_yahoo_ticker(self, pair: str) -> str:
        if pair in self.yahoo_map: return self.yahoo_map[pair]
        working_pair = pair
        if "/" in working_pair:
            base, quote = working_pair.split('/')
            if base in ['XBT', 'XXBT']: base = 'BTC'
            if base == 'XETH': base = 'ETH'
            if quote in ['ZEUR', 'ZEUR']: quote = 'EUR'
            if quote in ['ZUSD', 'ZUSD']: quote = 'USD'
            return f"{base}-{quote}"
        return working_pair

    def _parse_timedelta(self, duration_str: str) -> Optional[datetime.timedelta]:
        if not duration_str or len(duration_str) < 2: return None
        unit = duration_str[-1].lower()
        try:
            value = int(duration_str[:-1])
        except ValueError: return None
        if unit == 'm': return datetime.timedelta(minutes=value)
        if unit == 'h': return datetime.timedelta(hours=value)
        if unit == 'd': return datetime.timedelta(days=value)
        if unit == 'w': return datetime.timedelta(weeks=value)
        return None

    # ==============================================================================
    # FUNZIONE PER IL CALCOLO INDICATORI (Chiamata dentro getCandles)
    # ==============================================================================
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcola EMA, RSI e ATR aggiungendo le colonne al DataFrame.
        """
        if df.empty: return df

        # Assicuriamoci che i dati siano ordinati per data
        df = df.sort_index()

        # 1. EMA (Exponential Moving Average)
        df['EMA_Fast'] = df['Close'].ewm(span=self.ema_fast_span, adjust=False).mean()
        df['EMA_Slow'] = df['Close'].ewm(span=self.ema_slow_span, adjust=False).mean()

        # 2. RSI (Relative Strength Index)
        delta = df['Close'].diff()

        # Separa guadagni e perdite
        gain = (delta.where(delta > 0, 0))
        loss = (-delta.where(delta < 0, 0))

        # Media Mobile Esponenziale (Standard per RSI)
        avg_gain = gain.ewm(span=self.rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(span=self.rsi_period, adjust=False).mean()

        # Calcolo RS e gestione divisione per zero
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df['RSI'] = 100 - (100 / (1 + rs))

        # Riempie i primi valori NaN (dove non c'è abbastanza storico) con 50 (neutro)
        df['RSI'] = df['RSI'].fillna(50)

        # 3. ATR (Average True Range)
        prev_close = df['Close'].shift(1)
        tr1 = df['High'] - df['Low']
        tr2 = (df['High'] - prev_close).abs()
        tr3 = (df['Low'] - prev_close).abs()

        # True Range è il massimo tra le tre differenze
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # ATR è la media mobile (spesso smoothed) del TR
        df['ATR'] = tr.ewm(span=self.atr_period, adjust=False).mean()
        df['ATR'] = df['ATR'].fillna(0)

        # Calcolo Spread (utile per info extra)
        df['Spread'] = df['High'] - df['Low']

        # Pulizia finale NaN
        df.fillna(0, inplace=True)

        return df

    def getCandles(self, pair: str, interval: str, range_period: str = "1mo", truncate_to: str = None) -> List[Dict[str, Any]]:
        """
        Recupera le candele (Yahoo o Kraken), calcola gli indicatori e restituisce una lista di dizionari.
        """
        # Se richiesto 'now', usa la funzione specifica (che ritorna RSI/ATR vuoti o approssimati)
        if interval.lower() == 'now':
            return self._fetch_kraken_now(pair)

        yf_ticker = self._get_yahoo_ticker(pair)
        data = pd.DataFrame()

        # Tentativo 1: Yahoo Finance
        if yf_ticker and interval.lower() == '1d':
            try:
                yf_interval = self._normalize_interval_yahoo(interval)
                if yf_interval:
                    print(f"[INFO] Recupero dati Yahoo Finance per {pair} ({yf_ticker}) intervallo {yf_interval}... periodo: {range_period}")
                    df = yf.download(tickers=yf_ticker, period=range_period, interval=yf_interval, auto_adjust=True, progress=False, multi_level_index=False)
                    if not df.empty:
                        data = df
                        if 'Close' not in data.columns and 'Price' in data.columns:
                            data.rename(columns={'Price': 'Close'}, inplace=True)
                        if interval == '4h' and yf_interval == '1h':
                            data = self._resample_data(data, '4h')
            except Exception:
                pass

        # Tentativo 2: Kraken History
        if data.empty:
            data = self._fetch_kraken_history(pair, interval)

        if data.empty:
            print(f"[ERROR] Nessun dato trovato per {pair}")
            return []

        # --- QUI CHIAMIAMO LA FUNZIONE PER POPOLARE RSI E ATR ---
        data = self._calculate_indicators(data)

        data.reset_index(inplace=True)
        col_map = {'Date': 'timestamp', 'Datetime': 'timestamp', 'index': 'timestamp'}
        data.rename(columns=col_map, inplace=True)

        # Troncamento temporale (opzionale)
        if truncate_to:
            delta = self._parse_timedelta(truncate_to)
            if delta:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                if data['timestamp'].dt.tz is None:
                    data['timestamp'] = data['timestamp'].dt.tz_localize('UTC')
                else:
                    data['timestamp'] = data['timestamp'].dt.tz_convert('UTC')
                cutoff_time = pd.Timestamp.now(tz='UTC') - delta
                data = data[data['timestamp'] >= cutoff_time]

        data['timestamp'] = data['timestamp'].astype(str)
        result = data.to_dict(orient='records')

        final_list = []
        for row in result:
            # Converte tutte le chiavi in minuscolo (RSI -> rsi, ATR -> atr)
            clean_row = {k.lower(): v for k, v in row.items()}

            # Assegna esplicitamente il timeframe
            clean_row['timeframe'] = interval

            # Gestione fallback se mancano bid/ask/mid
            if 'bid' not in clean_row or clean_row['bid'] is None:
                close_p = clean_row.get('close', 0)
                high_p = clean_row.get('high', 0)
                low_p = clean_row.get('low', 0)
                clean_row['bid'] = close_p
                clean_row['ask'] = close_p
                clean_row['last'] = close_p
                clean_row['mid'] = (high_p + low_p) / 2 if (high_p and low_p) else close_p

            # Assicuriamoci che rsi e atr esistano nel dizionario finale (anche se 0)
            if 'rsi' not in clean_row: clean_row['rsi'] = 50.0
            if 'atr' not in clean_row: clean_row['atr'] = 0.0

            final_list.append(clean_row)

        return final_list

    def _resample_data(self, df: pd.DataFrame, target_rule: str) -> pd.DataFrame:
        if df.empty: return df
        agg_dict = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
        try:
            resampled = df.resample(target_rule).agg(agg_dict)
            resampled.dropna(inplace=True)
            return resampled
        except Exception:
            return df

    def _fetch_kraken_now(self, pair: str) -> List[Dict[str, Any]]:
        """
        Restituisce il prezzo attuale istantaneo.
        Nota: RSI e ATR sono impostati a None perché richiedono storico.
        """
        try:
            kraken_pair_id = self._get_kraken_id(pair)
            response = self.k.query_public('Ticker', {'pair': kraken_pair_id})
            if response.get('error'): return []
            results = response['result']
            data_key = list(results.keys())[0]
            data = results[data_key]
            def get_val(key, idx=0):
                val = data.get(key)
                if isinstance(val, list): return float(val[idx])
                return float(val)
            current_price = get_val('c', 0)
            ask_price = get_val('a', 0)
            bid_price = get_val('b', 0)
            open_price = get_val('o', 0)
            high_price = get_val('h', 0)
            low_price = get_val('l', 0)
            volume = get_val('v', 0)
            mid_price = (ask_price + bid_price) / 2 if (ask_price and bid_price) else current_price

            return [{
                'timestamp': datetime.datetime.now().isoformat(),
                'pair': pair,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': current_price,
                'volume': volume,
                'spread': ask_price - bid_price,
                'bid': bid_price,
                'ask': ask_price,
                'last': current_price,
                'mid': mid_price,
                'ema_fast': None,
                'ema_slow': None,
                'rsi': None,       # Non calcolabile su singolo tick
                'atr': None,       # Non calcolabile su singolo tick
                'timeframe': 'now'
            }]
        except Exception:
            return []

    def _fetch_kraken_history(self, pair: str, interval: str) -> pd.DataFrame:
        intervals_map = {'1m': 1, '5m': 5, '15m': 15, '30m': 30, '1h': 60, '4h': 240, '1d': 1440}
        mins = intervals_map.get(interval, 60)
        try:
            kraken_pair_id = self._get_kraken_id(pair)
            response = self.k.query_public('OHLC', {'pair': kraken_pair_id, 'interval': mins})
            if response.get('error'): return pd.DataFrame()
            res_data = response['result']
            data_key = list(res_data.keys())[0]
            ohlc_list = res_data[data_key]
            df = pd.DataFrame(ohlc_list, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'vwap', 'Volume', 'count'])
            cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            df[cols] = df[cols].astype(float)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df.set_index('timestamp', inplace=True)

            # --- MODIFICA AGGIORNATA: RESAMPLING COMPLETO ---
            if not df.empty:
                resample_rule = f"{mins}T"

                # 1. Creiamo un nuovo DataFrame con tutti i buchi riempiti di NaN (resample)
                # Includiamo 'count' e 'vwap' per completezza, anche se non usati dal DB
                df_resampled = df.resample(resample_rule).agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum',
                    'count': 'sum',
                    'vwap': 'last' # Vwap teorico dell'ultimo trade
                })

                # 2. RIEMPIMENTO "BUCHI" (Candele piatte)
                # Se un minuto non ha scambi, Open=High=Low=Close = Close precedente

                # Forward fill su Close: il prezzo rimane quello dell'ultima chiusura nota
                df_resampled['Close'] = df_resampled['Close'].ffill()

                # Le altre colonne di prezzo prendono il valore di Close appena calcolato
                df_resampled['Open'] = df_resampled['Open'].fillna(df_resampled['Close'])
                df_resampled['High'] = df_resampled['High'].fillna(df_resampled['Close'])
                df_resampled['Low'] = df_resampled['Low'].fillna(df_resampled['Close'])
                df_resampled['vwap'] = df_resampled['vwap'].fillna(df_resampled['Close'])

                # Volume e Count a 0 dove mancavano dati
                df_resampled['Volume'] = df_resampled['Volume'].fillna(0)
                df_resampled['count'] = df_resampled['count'].fillna(0)

                # Rimuoviamo eventuali righe iniziali che sono ancora NaN (se il primo dato manca)
                df = df_resampled.dropna()

            return df
        except Exception:
            return pd.DataFrame()

    def _normalize_interval_yahoo(self, interval: str) -> Optional[str]:
        valid = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']
        if interval in valid: return interval
        if interval == '4h': return '1h'
        if interval == '24h': return '1d'
        return '1d'

    def merge_candles_data(self, *candles_lists: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not candles_lists: return []
        analyzed_lists = []
        for clist in candles_lists:
            if not clist: continue
            granularity = 0.0
            if len(clist) >= 2:
                try:
                    t1 = pd.to_datetime(clist[0]['timestamp'])
                    t2 = pd.to_datetime(clist[1]['timestamp'])
                    granularity = abs((t2 - t1).total_seconds())
                except: pass
            analyzed_lists.append((granularity, clist))

        analyzed_lists.sort(key=lambda x: x[0])
        merged_dict = {}
        for granularity, clist in analyzed_lists:
            for candle in clist:
                merged_dict[candle['timestamp']] = candle

        result = list(merged_dict.values())
        result.sort(key=lambda x: x['timestamp'])
        return result

    def print_candles(self, candles: List[Dict[str, Any]]):
        if not candles:
            print(" -> Nessun dato da visualizzare.")
            return

        GREEN = '\033[92m'
        RED = '\033[91m'
        BOLD = '\033[1m'
        RESET = '\033[0m'
        CYAN = '\033[96m'
        YELLOW = '\033[93m'

        header = f"{'TIMESTAMP':<20} {'OPEN':<9} {'HIGH':<9} {'LOW':<9} {'CLOSE':<9} {'RSI':<6} {'ATR':<8} {'VOLUME':<12}"
        print(f"\n{BOLD}{header}{RESET}")
        print("-" * 100)

        for c in candles:
            ts = str(c.get('timestamp', ''))[:19].replace('T', ' ')
            o, h, l, cl = c.get('open', 0), c.get('high', 0), c.get('low', 0), c.get('close', 0)
            vol = c.get('volume', 0)
            rsi = c.get('rsi', 0)
            atr = c.get('atr', 0)

            def fmt(v): return f"{v:.2f}" if v and v > 10 else (f"{v:.4f}" if v else "0.00")
            row_col = GREEN if cl >= o else RED

            # Colora RSI
            rsi_val = rsi if rsi else 50
            rsi_col = RED if rsi_val > 70 else (GREEN if rsi_val < 30 else RESET)

            print(f"{ts:<20} {fmt(o):<9} {fmt(h):<9} {fmt(l):<9} {row_col}{fmt(cl):<9}{RESET} {rsi_col}{rsi_val:.1f}{RESET}   {fmt(atr):<8} {fmt(vol):<12}")
        print("-" * 100 + "\n")

    def getAllPairs(self, quote_filter: str = "EUR", leverage_only: bool = False) -> List[Dict[str, Any]]:
        if not self.pair_metadata: self._load_kraken_asset_pairs()
        all_pairs = []
        seen_ids = set()
        for k_id in self.pair_metadata:
            if k_id in seen_ids: continue
            try:
                pair_data = self.getPair(k_id)
                if pair_data['base'] == "N/A" or pair_data['quote'] == "N/A": continue
                if quote_filter and pair_data['quote'] != quote_filter: continue
                if leverage_only:
                    limits = pair_data.get('pair_limits', {})
                    if not (limits.get('can_leverage_buy') and limits.get('can_leverage_sell')): continue
                all_pairs.append(pair_data)
                seen_ids.add(k_id)
            except Exception: continue
        return all_pairs

    # ==========================================================================================
    # BINANCE WS (TRADES + DEPTH10) PER INSERIMENTO SU DB
    # ==========================================================================================
    async def stream_binance_market(self, pair: str, max_ticks: int = 50, depth_levels: int = 10) -> Dict[str, Any]:
        """
        Sottoscrive il WebSocket pubblico Binance (trade + depth10@100ms) per una singola coppia.
        Ritorna una struttura pronta per l'inserimento a DB con:
          - ticks: lista di trade normalizzati
          - orderbook: snapshot depth (top N livelli)

        NOTE:
          - Richiede la libreria 'websockets' (pip install websockets)
          - Il book restituito è l'ultimo update ricevuto durante la sessione
          - Non fa persistence: pensato per essere richiamato dal collector che poi salva su DB
        """
        if websockets is None:
            raise ImportError("La libreria 'websockets' non è installata. Esegui: pip install websockets")

        symbol = pair.replace("/", "").lower()  # es. ETHEUR -> etheur
        stream = f"{symbol}@trade/{symbol}@depth{depth_levels}@100ms"
        url = f"wss://stream.binance.com:9443/stream?streams={stream}"

        ticks: List[Dict[str, Any]] = []
        last_orderbook: Optional[Dict[str, Any]] = None

        # Per evitare loop infiniti, interrompiamo dopo max_ticks trades raccolti
        async with websockets.connect(url, ping_interval=20, close_timeout=5) as ws:
            while len(ticks) < max_ticks:
                raw = await ws.recv()
                message = json.loads(raw)
                stream_name = message.get("stream", "")
                payload = message.get("data", {})

                if stream_name.endswith("@trade"):
                    # Trade event
                    trade_ts = datetime.datetime.utcfromtimestamp(payload.get("E", 0) / 1000).isoformat()
                    ticks.append({
                        "pair": pair,
                        "timestamp": trade_ts,
                        "price": float(payload.get("p", 0)),
                        "size": float(payload.get("q", 0)),
                        "side": "SELL" if payload.get("m") else "BUY",
                        "trade_id": payload.get("t"),
                        "exchange": "binance"
                    })
                elif "@depth" in stream_name:
                    # Depth update (solo top N livelli)
                    depth_ts = datetime.datetime.utcfromtimestamp(payload.get("E", 0) / 1000).isoformat()
                    bids = payload.get("b", [])[:depth_levels]
                    asks = payload.get("a", [])[:depth_levels]

                    orderbook_rows: List[Dict[str, Any]] = []
                    for idx, b in enumerate(bids):
                        orderbook_rows.append({
                            "pair": pair,
                            "timestamp": depth_ts,
                            "side": "BID",
                            "price": float(b[0]),
                            "size": float(b[1]),
                            "depth_rank": idx + 1,
                            "exchange": "binance"
                        })
                    for idx, a in enumerate(asks):
                        orderbook_rows.append({
                            "pair": pair,
                            "timestamp": depth_ts,
                            "side": "ASK",
                            "price": float(a[0]),
                            "size": float(a[1]),
                            "depth_rank": idx + 1,
                            "exchange": "binance"
                        })

                    last_orderbook = {
                        "pair": pair,
                        "timestamp": depth_ts,
                        "exchange": "binance",
                        "levels": orderbook_rows
                    }

        return {
            "pair": pair,
            "ticks": ticks,
            "orderbook": last_orderbook
        }

if __name__ == "__main__":
    provider = MarketDataProvider()

    print("--- SCARICO DATI ---")
    all_pairs = provider.getAllPairs(quote_filter="EUR", leverage_only=True)
    print(f"Trovate {len(all_pairs)} coppie totali su Kraken.")

    # Stampa le prime 3 per esempio
    for p in all_pairs:
        # print(f" - {p['pair']} (ID: {p['kr_pair']}) -> Min Order: {p['pair_limits']['ordermin']}")
        # data_1d = provider.getCandles(p['pair'], "1d", "1mo", truncate_to="30d")
        # data_4h = provider.getCandles(p['pair'], "4h", "5d", truncate_to="5d")
        # data_1h = provider.getCandles(p['pair'], "1h", "1d", truncate_to="6h")
        # data_15m = provider.getCandles(p['pair'], "15m", "1d", truncate_to="2h")
        # data_5m = provider.getCandles(p['pair'], "5m", "1d", truncate_to="30m")
        # data_now = provider.getCandles(p['pair'], "now")
        # merged_data = provider.merge_candles_data(data_1d, data_4h, data_1h, data_15m, data_5m, data_now)
        # provider.print_candles(merged_data)
        # print(f" - {p['pair']} (ID: {p['kr_pair']}) -> Min Order: {p['pair_limits']['ordermin']}")
        print(f" - {p['pair']} (ID: {p['kr_pair']}) -> Min Order: {p['pair_limits']['ordermin']}")
        data_1m = provider.getCandles(p['pair'], "1m", "1d")
        provider.print_candles(data_1m)

