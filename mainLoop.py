import time
import threading
from datetime import datetime, timedelta

from db.DatabaseManager import DatabaseManager
from db.MarketDataProvider import MarketDataProvider
from db.TimeSfmForecaster import TimeSfmForecaster

# --- CONFIGURAZIONE BASE -----------------------------------------------------

QUOTE_FILTER = "EUR"
LEVERAGE_ONLY = True

# Per ogni timeframe, quanto storico chiedere a getCandles (Kraken API)
TIMEFRAME_CONFIG = {
    "1d":  "5d",
    "4h":  "2d",
    "1h":  "2d",
    "15m": "2d",
    "5m":  "1d",
    "1m":  "1d",
}

# Configurazione Forecaster
FORECAST_CONTEXT_LEN = 120  # Numero di candele storiche da passare al modello
FORECAST_STEPS = 3          # Quante candele future predire (+1, +2, +3)

# Lock per evitare che partano due forecast sovrapposti sulla GPU/CPU
forecast_lock = threading.Lock()

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
    Scarica i dati da Kraken e li salva nel DB.
    """
    market_prov = MarketDataProvider()
    db = DatabaseManager()

    try:
        for p in pairs:
            pair_name = p["pair"]

            for tf in timeframes:
                if tf not in TIMEFRAME_CONFIG:
                    continue

                lookback = TIMEFRAME_CONFIG[tf]
                try:
                    candles = market_prov.getCandles(pair_name, tf, lookback)
                    db.insert_currency_data(candles, p, "currency")
                except Exception as e:
                    print(f"[ERR] Fetch {pair_name} {tf}: {e}")

    finally:
        db.close_connection()


# --- JOB FORECAST (TimeSFM) & WRAPPER ASINCRONO -----------------------------

def job_forecast(pairs, forecaster: TimeSfmForecaster, tf):
    """
    Job che esegue il calcolo pesante.
    Viene chiamato dal thread background.
    """
    print(f"[{datetime.now()}] >>> INIZIO CALCOLO PREVISIONI {tf} (Thread)...")
    db = DatabaseManager()
    count_ok = 0

    try:
        for p in pairs:
            base_currency = p.get('base')

            # 1. Recupera storico dal DB (veloce, solo ultime 120 righe)
            history_data = db.get_last_candles('currency', tf, base_currency, FORECAST_CONTEXT_LEN)

            # Se non abbiamo abbastanza dati, saltiamo
            if not history_data or len(history_data) < 30:
                continue

            try:
                # 2. Genera Previsione (+1, +2, +3)
                forecast_rows = forecaster.predict_candles(
                    history_data,
                    tf,
                    FORECAST_STEPS,
                    p
                )

                # 3. Salva nel DB (tabella 'forecast')
                if forecast_rows:
                    db.insert_currency_data(forecast_rows, p, "forecast")
                    count_ok += 1

            except Exception as e:
                print(f"[ERR] Forecast {base_currency} {tf}: {e}")

    finally:
        db.close_connection()

    print(f"[{datetime.now()}] <<< FINE CALCOLO PREVISIONI {tf}. Generate: {count_ok}")


def run_forecast_in_background(pairs, forecaster, tf):
    """
    Wrapper che gestisce il Lock e lancia il job vero e proprio.
    """
    # Tenta di acquisire il lock senza bloccare.
    # Se è False, significa che un altro forecast è ancora in corso -> SKIP.
    if not forecast_lock.acquire(blocking=False):
        print(f"[{datetime.now()}] [WARN] Forecast {tf} SKIPPATO: Il precedente è ancora in esecuzione.")
        return

    try:
        job_forecast(pairs, forecaster, tf)
    except Exception as e:
        print(f"[ERR] Errore critico nel thread forecast: {e}")
    finally:
        forecast_lock.release()


# --- JOB SPECIFICI (5m, 15m, ecc.) ------------------------------------------

def job_5m(pairs):
    print(f"[{datetime.now()}] Avvio job_5m...")
    fetch_and_store_for_timeframes(pairs, ["5m", "1m"])
    print(f"[{datetime.now()}] Fine job_5m.")


def job_15m(pairs, forecaster):
    print(f"[{datetime.now()}] Avvio job_15m (Download dati)...")

    # 1. Scarica dati freschi (Bloccante, ma veloce ~secondi)
    fetch_and_store_for_timeframes(pairs, ["15m"])

    # 2. Avvia Forecast in BACKGROUND (Non bloccante)
    print(f"[{datetime.now()}] Avvio Thread Forecast 15m...")
    t = threading.Thread(target=run_forecast_in_background, args=(pairs, forecaster, "15m"))
    t.start()

    print(f"[{datetime.now()}] Fine job_15m (Main Thread libero).")


def job_1h_block(pairs, forecaster):
    print(f"[{datetime.now()}] Avvio job_1h_block (Download 1d, 4h, 1h)...")

    # 1. Scarica dati freschi
    fetch_and_store_for_timeframes(pairs, ["1d", "4h", "1h"])

    # 2. Avvia Forecast in BACKGROUND (Non bloccante) per 1h
    # (Se vuoi fare forecast anche per 4h e 1d, puoi lanciare altri thread o fare un ciclo nel job_forecast)
    print(f"[{datetime.now()}] Avvio Thread Forecast 1h...")
    t = threading.Thread(target=run_forecast_in_background, args=(pairs, forecaster, "1h"))
    t.start()

    print(f"[{datetime.now()}] Fine job_1h_block (Main Thread libero).")


# --- LOOP PRINCIPALE --------------------------------------------------------

def main_loop():
    print("=== AVVIO COLLECTOR CONTINUO (ASYNC FORECAST) ===")

    # 1. Inizializzazione Risorse
    market_prov = MarketDataProvider()
    pairs = load_pairs(market_prov)

    print("=== Inizializzazione TimeSfmForecaster (Heavy Load)... ===")
    forecaster = TimeSfmForecaster()
    print("=== TimeSfmForecaster Pronto ===")

    # Timer
    last_5m_run = None
    last_15m_run = None
    last_1h_block_run = None

    while True:
        now = datetime.now()

        # Job ogni 5 minuti
        if last_5m_run is None or (now - last_5m_run) >= timedelta(minutes=5):
            job_5m(pairs)
            last_5m_run = now

        # Job ogni 15 minuti (Include Forecast Async)
        if last_15m_run is None or (now - last_15m_run) >= timedelta(minutes=15):
            job_15m(pairs, forecaster)
            last_15m_run = now

        # Job ogni 60 minuti (Include Forecast Async)
        if last_1h_block_run is None or (now - last_1h_block_run) >= timedelta(hours=1):
            job_1h_block(pairs, forecaster)
            last_1h_block_run = now

        # Piccola pausa per risparmio CPU
        time.sleep(10)


if __name__ == "__main__":
    main_loop()
