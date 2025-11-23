import os
import time
import datetime
import pandas as pd
import yfinance as yf
import krakenex
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

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

        self.ema_fast_span = 12
        self.ema_slow_span = 26

    def _load_kraken_asset_pairs(self):
        try:
            response = self.k.query_public('AssetPairs')
            if response.get('error'): return

            results = response['result']
            for pair_id, info in results.items():
                wsname = info.get('wsname')
                altname = info.get('altname')

                self.kraken_pairs_map[pair_id] = pair_id
                if wsname: self.kraken_pairs_map[wsname] = pair_id
                if altname: self.kraken_pairs_map[altname] = pair_id

                self.pair_metadata[pair_id] = {
                    'base': info.get('base'),
                    'quote': info.get('quote'),
                    'wsname': wsname,
                    'altname': altname
                }
        except Exception:
            pass

    def _get_kraken_id(self, pair: str) -> str:
        return self.kraken_pairs_map.get(pair, pair)

    def getPair(self, pair: str) -> Dict[str, str]:
        k_id = self._get_kraken_id(pair)
        meta = self.pair_metadata.get(k_id, {})

        base_clean = "N/A"
        quote_clean = "N/A"
        human_pair = meta.get('wsname', pair)

        if human_pair and '/' in human_pair:
            base_clean, quote_clean = human_pair.split('/')
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

            if base_clean and quote_clean:
                human_pair = f"{base_clean}/{quote_clean}"

        return {
            "base": base_clean,
            "quote": quote_clean,
            "pair": human_pair,
            "kr_pair": k_id
        }

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
        """Converte stringhe come '1h', '30m', '1d' in timedelta."""
        if not duration_str or len(duration_str) < 2:
            return None

        unit = duration_str[-1].lower()
        try:
            value = int(duration_str[:-1])
        except ValueError:
            return None

        if unit == 'm': return datetime.timedelta(minutes=value)
        if unit == 'h': return datetime.timedelta(hours=value)
        if unit == 'd': return datetime.timedelta(days=value)
        if unit == 'w': return datetime.timedelta(weeks=value)
        return None

    def getCandles(self, pair: str, interval: str, range_period: str = "1mo", truncate_to: str = None) -> List[Dict[str, Any]]:
        """
        Recupera le candele.
        :param truncate_to: (Opzionale) Stringa es. '4h', '12h'. Se presente, taglia i risultati
                            mantenendo solo quelli più recenti di questo lasso di tempo.
        """

        # CASO 1: REAL TIME
        if interval.lower() == 'now':
            return self._fetch_kraken_now(pair)

        # CASO 2: STORICO
        yf_ticker = self._get_yahoo_ticker(pair)
        data = pd.DataFrame()

        if yf_ticker:
            try:
                yf_interval = self._normalize_interval_yahoo(interval)
                if yf_interval:
                    df = yf.download(tickers=yf_ticker, period=range_period, interval=yf_interval, auto_adjust=True, progress=False, multi_level_index=False)
                    if not df.empty:
                        data = df
                        if 'Close' not in data.columns and 'Price' in data.columns:
                            data.rename(columns={'Price': 'Close'}, inplace=True)

                        if interval == '4h' and yf_interval == '1h':
                            data = self._resample_data(data, '4h')
            except Exception:
                pass

        if data.empty:
            data = self._fetch_kraken_history(pair, interval)

        if data.empty:
            print(f"[ERROR] Nessun dato trovato per {pair}")
            return []

        data = self._calculate_indicators(data)
        data.reset_index(inplace=True)

        col_map = {'Date': 'timestamp', 'Datetime': 'timestamp', 'index': 'timestamp'}
        data.rename(columns=col_map, inplace=True)

        # --- NUOVA LOGICA TRUNCATE (Sul DataFrame) ---
        if truncate_to:
            delta = self._parse_timedelta(truncate_to)
            if delta:
                # 1. Assicuriamoci che la colonna sia datetime
                data['timestamp'] = pd.to_datetime(data['timestamp'])

                # 2. Standardizzazione Timezone a UTC per confronto
                if data['timestamp'].dt.tz is None:
                    data['timestamp'] = data['timestamp'].dt.tz_localize('UTC')
                else:
                    data['timestamp'] = data['timestamp'].dt.tz_convert('UTC')

                # 3. Calcolo Cutoff
                cutoff_time = pd.Timestamp.now(tz='UTC') - delta

                # 4. Filtro DataFrame
                data = data[data['timestamp'] >= cutoff_time]

        # Converti a stringa per output finale
        data['timestamp'] = data['timestamp'].astype(str)

        result = data.to_dict(orient='records')

        final_list = []
        for row in result:
            clean_row = {k.lower(): v for k, v in row.items()}

            # Popolamento dati mancanti
            if 'bid' not in clean_row or clean_row['bid'] is None:
                close_p = clean_row.get('close', 0)
                high_p = clean_row.get('high', 0)
                low_p = clean_row.get('low', 0)
                clean_row['bid'] = close_p
                clean_row['ask'] = close_p
                clean_row['last'] = close_p
                if high_p and low_p:
                    clean_row['mid'] = (high_p + low_p) / 2
                else:
                    clean_row['mid'] = close_p

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
                'ema_slow': None
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
            return df
        except Exception:
            return pd.DataFrame()

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty: return df
        df['Spread'] = df['High'] - df['Low']
        df['EMA_Fast'] = df['Close'].ewm(span=self.ema_fast_span, adjust=False).mean()
        df['EMA_Slow'] = df['Close'].ewm(span=self.ema_slow_span, adjust=False).mean()
        df.fillna(0, inplace=True)
        return df

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
            granularity = float('inf')
            if len(clist) >= 2:
                try:
                    t1 = pd.to_datetime(clist[0]['timestamp'])
                    t2 = pd.to_datetime(clist[1]['timestamp'])
                    granularity = abs((t2 - t1).total_seconds())
                except: pass
            else:
                granularity = 0.0
            analyzed_lists.append((granularity, clist))

        # Ordina per granularità crescente (il più grande vince)
        analyzed_lists.sort(key=lambda x: x[0], reverse=False)

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

        header = f"{'TIMESTAMP':<20} {'OPEN':<10} {'HIGH':<10} {'LOW':<10} {'CLOSE':<10} {'VOLUME':<15} {'BID':<10} {'ASK':<10} {'MID':<10} {'EMA F/S':<15}"
        print(f"\n{BOLD}{header}{RESET}")
        print("-" * 150)

        for c in candles:
            ts = str(c.get('timestamp', ''))[:19].replace('T', ' ')
            o = c.get('open', 0)
            h = c.get('high', 0)
            l = c.get('low', 0)
            cl = c.get('close', 0)
            vol = c.get('volume', 0)

            bid = c.get('bid')
            ask = c.get('ask')
            mid = c.get('mid')

            ef, es = c.get('ema_fast'), c.get('ema_slow')

            def fmt(v, is_vol=False):
                if v is None: return "-"
                if is_vol:
                    if v > 1000: return f"{v:,.2f}"
                    return f"{v:.4f}"
                return f"{v:.2f}" if v > 10 else f"{v:.5f}"

            row_col = GREEN if cl >= o else RED

            ema_str = "N/A"
            if ef and es:
                ecol = CYAN if ef > es else RESET
                ema_str = f"{ecol}{fmt(ef)}/{fmt(es)}{RESET}"

            print(f"{ts:<20} {fmt(o):<10} {fmt(h):<10} {fmt(l):<10} {row_col}{fmt(cl):<10}{RESET} {fmt(vol, True):<15} {fmt(bid):<10} {fmt(ask):<10} {fmt(mid):<10} {ema_str}")
        print("-" * 150 + "\n")

if __name__ == "__main__":
    provider = MarketDataProvider()

    print("--- SCARICO DATI ---")
    # Nota: Yahoo 15m su lunghi periodi (1mo) spesso fallisce o limita a 60gg.
    # Per 4h su 5gg ora dovresti vedere i salti di 4 ore nei timestamp
    data_1d = provider.getCandles("XBT/EUR", "1d", "1mo")
    data_4h = provider.getCandles("XBT/EUR", "4h", "5d")
    data_1h = provider.getCandles("XBT/EUR", "1h", "1d", truncate_to="6h")
    data_15m = provider.getCandles("XBT/EUR", "15m", "1d", truncate_to="2h")
    data_5m = provider.getCandles("XBT/EUR", "5m", "1d", truncate_to="30m")
    data_now = provider.getCandles("XBT/EUR", "now")

    print(f"1D Candles: {len(data_1d)}")
    print(f"4H Candles (Resampled): {len(data_4h)}") # Dovrebbero essere circa 30 (5gg * 6 candele)
    print(f"1H Candles: {len(data_1h)}")
    print(f"15M Candles: {len(data_15m)}")

    print("\n--- MERGE (Priority: Now > 15m > 1h > 4h > 1d) ---")
    merged_data = provider.merge_candles_data(data_1d, data_4h, data_1h, data_15m, data_5m, data_now)

    print(f"Total Merged: {len(merged_data)}")

    print("\n--- ULTIMI 20 RECORD MERGED ---")
    provider.print_candles(merged_data)

    print("\n--- TEST GET PAIR ---")
    print(provider.getPair("BTC/EUR"))  # Deve dare BTC/EUR, XXBTZEUR
    print(provider.getPair("YFI/EUR"))   # Deve dare YFI/EUR, YFIEUR (se esiste)

