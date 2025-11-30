import time
from datetime import datetime, timedelta

from db.DatabaseManager import DatabaseManager
from db.MarketDataProvider import MarketDataProvider


# --- CONFIGURAZIONE BASE -----------------------------------------------------

QUOTE_FILTER = "EUR"
LEVERAGE_ONLY = True

# Per ogni timeframe, quanto storico chiedere a getCandles
TIMEFRAME_CONFIG = {
    "1d":  "5d",
    "4h":  "5d",
    "1h":  "5d",
    "15m": "1d",
    "5m":  "1d",
    "1m":  "1d",
}


# --- FUNZIONI DI SUPPORTO ----------------------------------------------------

def load_pairs(market_prov: MarketDataProvider):
    """
    Carica una volta tutte le coppie con quote EUR e leva.
    """
    return market_prov.getAllPairs(
        quote_filter=QUOTE_FILTER,
        leverage_only=LEVERAGE_ONLY
    )


def fetch_and_store_for_timeframes(pairs, timeframes):
    """
    Fa quello che fa main3 ma solo per i timeframe indicati.
    - pairs: lista di dict con almeno 'pair'
    - timeframes: lista/tupla di stringhe (es. ["5m"] oppure ["15m", "5m"])
    """
    market_prov = MarketDataProvider()
    db = DatabaseManager()

    try:
        for p in pairs:
            pair_name = p["pair"]

            for tf in timeframes:
                if tf not in TIMEFRAME_CONFIG:
                    continue  # se dimentichi di configurare il timeframe, lo salta

                lookback = TIMEFRAME_CONFIG[tf]
                candles = market_prov.getCandles(pair_name, tf, lookback)
                db.insert_currency_data(candles, p, "currency")

    finally:
        db.close_connection()


# --- JOB SPECIFICI (5m, 15m, ecc.) ------------------------------------------

def job_5m(pairs):
    """
    Job che gira ogni 5 minuti.
    Qui prendo solo il timeframe 5m (puoi aggiungerne altri se vuoi).
    """
    print(f"[{datetime.now()}] Avvio job_5m...")
    fetch_and_store_for_timeframes(pairs, ["5m", "1m"])
    print(f"[{datetime.now()}] Fine job_5m.")


def job_15m(pairs):
    """
    Job che gira ogni 15 minuti.
    Esempio: qui prendo 15m e, se vuoi, anche 1m/5m.
    """
    print(f"[{datetime.now()}] Avvio job_15m...")
    fetch_and_store_for_timeframes(pairs, ["15m"])
    print(f"[{datetime.now()}] Fine job_15m.")


# Se vuoi anche qualcosa a 1h, 4h, 1d, puoi definirlo così:
def job_1h_block(pairs):
    """
    Esempio di job “più pesante” ogni ora:
    riprende la logica di main3 su tutti i timeframe principali.
    """
    print(f"[{datetime.now()}] Avvio job_1h_block (1d,4h,1h,15m,5m,1m)...")
    fetch_and_store_for_timeframes(pairs, ["1d", "4h", "1h", "15m"])
    print(f"[{datetime.now()}] Fine job_1h_block.")


# --- LOOP PRINCIPALE --------------------------------------------------------

def main_loop():
    """
    Loop infinito che esegue:
      - job_5m ogni 5 minuti
      - job_15m ogni 15 minuti
      - job_1h_block ogni 60 minuti (opzionale, puoi commentarlo)
    """

    print("=== AVVIO COLLECTOR CONTINUO ===")

    market_prov = MarketDataProvider()
    pairs = load_pairs(market_prov)

    # Tieni traccia dell’ultima esecuzione di ciascun job
    last_5m_run = None
    last_15m_run = None
    last_1h_block_run = None

    while True:
        now = datetime.now()

        # Job ogni 5 minuti
        if last_5m_run is None or (now - last_5m_run) >= timedelta(minutes=5):
            job_5m(pairs)
            last_5m_run = now

        # Job ogni 15 minuti
        if last_15m_run is None or (now - last_15m_run) >= timedelta(minutes=15):
            job_15m(pairs)
            last_15m_run = now

        # Job “blocco grosso” ogni 60 minuti (opzionale)
        if last_1h_block_run is None or (now - last_1h_block_run) >= timedelta(hours=1):
            job_1h_block(pairs)
            last_1h_block_run = now

        # Piccola pausa per non sforzare la CPU
        time.sleep(10)


if __name__ == "__main__":
    main_loop()
