import os
import time
import datetime
import pandas as pd
import yfinance as yf
import krakenex
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

class MarketDataProvider:
    def __init__(self, yahoo_map: Dict[str, str]):
        """
        Inizializza il provider dati.

        :param kraken_key: API Key Kraken
        :param kraken_secret: API Secret Kraken
        :param yahoo_map: Dizionario per mappare le pair Kraken su Yahoo.
                          Es: {'XBT/EUR': 'BTC-EUR', 'ETH/EUR': 'ETH-EUR'}
        """
        # Carica variabili env
        load_dotenv()

        kraken_key = os.getenv("KRAKEN_KEY")
        kraken_secret = os.getenv("KRAKEN_SECRET")
        self.k = krakenex.API(key=kraken_key, secret=kraken_secret)
        self.yahoo_map = yahoo_map

        # Configurazione EMA default
        self.ema_fast_span = 12
        self.ema_slow_span = 26

    def getCandles(self, pair: str, interval: str, range_period: str = "1mo") -> List[Dict[str, Any]]:
        """
        Recupera le candele (o il ticker attuale).

        :param pair: Nome della pair (es. 'XBT/EUR')
        :param interval: 'now', '1m', '5m', '15m', '30m', '1h', '1d', etc.
        :param range_period: '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', 'max'
        """

        # CASO 1: Dati in tempo reale (NOW)
        if interval.lower() == 'now':
            return self._fetch_kraken_now(pair)

        # Recupero simbolo Yahoo
        yf_ticker = self.yahoo_map.get(pair)

        data = pd.DataFrame()

        # CASO 2: Tentativo con Yahoo Finance (Prioritario)
        if yf_ticker:
            try:
                print(f"[INFO] Scarico dati da Yahoo per {yf_ticker}...")
                # Convertiamo l'intervallo per Yahoo (es. 4h -> non supportato bene da yf free, fallback su 1h e resample o kraken)
                yf_interval = self._normalize_interval_yahoo(interval)

                if yf_interval:
                    df = yf.download(
                        tickers=yf_ticker,
                        period=range_period,
                        interval=yf_interval,
                        auto_adjust=True,
                        progress=False,
                        multi_level_index=False # Importante per nuove versioni di yfinance
                    )

                    if not df.empty:
                        data = df
                        # Standardizzazione colonne (Yahoo a volte usa 'Close' o 'Adj Close')
                        if 'Close' not in data.columns and 'Price' in data.columns:
                            data.rename(columns={'Price': 'Close'}, inplace=True)
            except Exception as e:
                print(f"[WARN] Errore Yahoo Finance: {e}. Passo a Kraken.")

        # CASO 3: Fallback su Kraken (se Yahoo fallisce o manca mapping)
        if data.empty:
            print(f"[INFO] Recupero dati storici da Kraken per {pair}...")
            data = self._fetch_kraken_history(pair, interval)

        # Se ancora vuoto, errore
        if data.empty:
            print(f"[ERROR] Nessun dato trovato per {pair}")
            return []

        # Calcolo Indicatori
        data = self._calculate_indicators(data)

        # Conversione in lista di oggetti (dizionari)
        # Reset index per avere la data come colonna
        data.reset_index(inplace=True)

        # Rinomina colonna data/index in 'timestamp'
        col_map = {
            'Date': 'timestamp',
            'Datetime': 'timestamp',
            'index': 'timestamp'
        }
        data.rename(columns=col_map, inplace=True)

        # Conversione timestamp in stringa ISO o unix se necessario
        data['timestamp'] = data['timestamp'].astype(str)

        # Trasforma in lista di dict e normalizza chiavi in minuscolo
        result = data.to_dict(orient='records')
        result_clean = []

        for row in result:
            clean_row = {k.lower(): v for k, v in row.items()}
            result_clean.append(clean_row)

        return result_clean

    def _fetch_kraken_now(self, pair: str) -> List[Dict[str, Any]]:
        """Chiama il Ticker di Kraken per dati real-time"""
        try:
            # Bisogna convertire la pair nel formato API Kraken se necessario (es. XBT/EUR -> XXBTZEUR)
            # Qui assumiamo che 'pair' o la mappa siano gestiti, oppure usiamo AssetPairs per trovarlo.
            # Per semplicità, proviamo la query diretta, Kraken spesso accetta alias.

            response = self.k.query_public('Ticker', {'pair': pair})

            if response.get('error'):
                print(f"[ERROR] Kraken Ticker: {response['error']}")
                return []

            results = response['result']
            # Prende il primo (e unico) risultato
            pair_key = list(results.keys())[0]
            data = results[pair_key]

            # Parsing risposta Kraken Ticker
            # a = ask array(price, lot, lot), b = bid array, c = last trade closed array(price, lot)
            # v = volume array(today, 24h), o = open

            current_price = float(data['c'][0])
            ask_price = float(data['a'][0])
            bid_price = float(data['b'][0])
            open_price = float(data['o'][0]) # Open oggi
            high_price = float(data['h'][0]) # High oggi
            low_price = float(data['l'][0])  # Low oggi
            volume = float(data['v'][0])     # Volume oggi

            # Calcolo indicatori istantanei
            spread = ask_price - bid_price

            # Nota: EMA non calcolabile su un singolo punto "now" senza storico.
            # Se necessario, bisognerebbe scaricare storico + now. Qui ritorniamo None o 0.

            return [{
                'timestamp': datetime.datetime.now().isoformat(),
                'pair': pair,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': current_price,
                'volume': volume,
                'spread': spread,
                'ema_fast': None, # Non calcolabile istantaneamente
                'ema_slow': None
            }]

        except Exception as e:
            print(f"[ERROR] Errore fetch now Kraken: {e}")
            return []

    def _fetch_kraken_history(self, pair: str, interval: str) -> pd.DataFrame:
        """Recupera OHLC da Kraken"""

        # Mappa intervalli stringa -> minuti Kraken
        intervals_map = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '4h': 240, '1d': 1440, '1w': 10080, '15d': 21600
        }

        minute_interval = intervals_map.get(interval, 60) # Default 1h

        try:
            response = self.k.query_public('OHLC', {'pair': pair, 'interval': minute_interval})

            if response.get('error'):
                print(f"[ERROR] Kraken OHLC: {response['error']}")
                return pd.DataFrame()

            res_data = response['result']
            pair_key = list(res_data.keys())[0] # es. XXBTZEUR
            ohlc_list = res_data[pair_key]

            # Converti in DataFrame
            # Kraken OHLC: [time, open, high, low, close, vwap, volume, count]
            df = pd.DataFrame(ohlc_list, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'vwap', 'Volume', 'count'])

            # Pulizia tipi
            cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            df[cols] = df[cols].astype(float)

            # Conversione timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df.set_index('timestamp', inplace=True)

            return df

        except Exception as e:
            print(f"[ERROR] Errore storico Kraken: {e}")
            return pd.DataFrame()

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggiunge Spread e EMA al DataFrame"""
        if df.empty: return df

        # 1. Spread (Per storico usiamo High - Low come proxy di volatilità,
        #    o 0 se non abbiamo Bid/Ask storici)
        #    Yahoo non dà Bid/Ask storici gratuiti.
        df['Spread'] = df['High'] - df['Low']

        # 2. EMA
        df['EMA_Fast'] = df['Close'].ewm(span=self.ema_fast_span, adjust=False).mean()
        df['EMA_Slow'] = df['Close'].ewm(span=self.ema_slow_span, adjust=False).mean()

        # Gestione NaN iniziali (opzionale, li sostituisce con 0 o li lascia)
        df.fillna(0, inplace=True)

        return df

    def _normalize_interval_yahoo(self, interval: str) -> Optional[str]:
        """Converte o valida l'intervallo per Yahoo"""
        valid_yahoo = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']

        if interval in valid_yahoo:
            return interval

        # Mapping custom
        if interval == '4h': return '1h' # Yahoo non ha 4h, scarichiamo 1h e l'utente dovrà aggregare o accontentarsi
        if interval == '24h': return '1d'

        return '1d' # Default safe


if __name__ == "__main__":


    # Mappatura personalizzata Pairs (Kraken -> Yahoo)
    PAIRS_MAPPING = {
        "XBT/EUR": "BTC-EUR",
        "ETH/EUR": "ETH-EUR",
        "SOL/EUR": "SOL-EUR",
        "ADA/EUR": "ADA-EUR"
    }

    # 1. Inizializzazione
    provider = MarketDataProvider(PAIRS_MAPPING)

    # --- ESEMPIO 1: Dati Storici (Prova Yahoo, poi Kraken) ---
    print("\n--- ESEMPIO STORICO 1h (Ultimo Mese) ---")
    candles = provider.getCandles(
        pair="XBT/EUR",
        interval="1h",
        range_period="1mo"
    )

    # Stampa primi 2 risultati
    if candles:
        print(f"Trovate {len(candles)} candele.")
        print(candles[-1]) # Ultima candela (la più recente)
    else:
        print("Nessuna candela trovata.")


    # --- ESEMPIO 2: Dati Real-Time (Kraken Ticker) ---
    print("\n--- ESEMPIO REAL-TIME (NOW) ---")
    live_data = provider.getCandles(
        pair="XBT/EUR",
        interval="now",
        range_period="all" # ininfluente per 'now'
    )

    print(live_data)
