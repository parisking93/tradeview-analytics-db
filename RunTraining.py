import sys
import os
import random
from datetime import datetime, timedelta

# Path Setup
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from db.DatabaseManager import DatabaseManager
from db.MarketDataProvider import MarketDataProvider
from trading.Vectorizer import DataVectorizer, VectorizerConfig
from trading.TrmAgent import MultiTimeframeTRM
from trading.Trainer import TradingTrainer

def train_loop():
    # --- CONFIGURAZIONE ---
    PAIR_NAME = "ETH/EUR" # Alleniamo su ETH per iniziare
    CURRENCY = "ETH"
    TF_CONFIG = {"1d": 30, "4h": 50, "1h": 100}

    # Quanto guardare nel futuro per decidere se era un buon trade?
    LOOKAHEAD_STEPS = 48 # 48 ore (se usiamo candele 1h)

    EPOCHS = 1000 # Quanti esempi casuali mostrare al modello

    # --- SETUP ---
    db = DatabaseManager()

    # Recuperiamo info coppia per limiti
    market_prov = MarketDataProvider()
    pairs = market_prov.getAllPairs()
    pair_info = next((p for p in pairs if p['pair'] == PAIR_NAME), None)

    if not pair_info:
        print(f"Coppia {PAIR_NAME} non trovata.")
        return

    # Inizializza Vectorizer e Modello
    vec_config = VectorizerConfig(candle_history_config=TF_CONFIG)
    vectorizer = DataVectorizer(vec_config)

    # Importante: Inizializza il modello con le dimensioni corrette
    model = MultiTimeframeTRM(
        tf_configs=TF_CONFIG,
        input_size_per_candle=vectorizer.candle_dim,
        static_size=vectorizer.static_total_dim, # Ora include il +1 del wallet
        hidden_dim=256
    )

    trainer = TradingTrainer(model, db, vectorizer)

    print(f"--- INIZIO TRAINING SU {PAIR_NAME} ---")

    # --- TRAINING LOOP (RANDOM SAMPLING) ---
    # Invece di iterare sequenzialmente, prendiamo punti random nel tempo
    # Questo rompe le correlazioni temporali e stabilizza il training (i.i.d.)

    # 1. Troviamo il range di date disponibili
    # Query veloce per min e max date
    candles = db.select_all("currency", f"base='{CURRENCY}' AND timeframe='1h' ORDER BY timestamp DESC LIMIT 5000")
    if len(candles) < 200:
        print("Pochi dati per il training.")
        return

    valid_dates = [c['timestamp'] for c in candles]
    # Rimuoviamo le date troppo recenti (perché non avremmo il futuro per l'oracolo)
    # Assumiamo che valid_dates[0] sia la più recente.
    trainable_dates = valid_dates[LOOKAHEAD_STEPS:]

    moving_avg_loss = 0.0
    tf_config_forecast = {"1d+1": 1, "1d+2": 1, "4h+1": 1, "4h+2": 1}
    for i in range(EPOCHS):
        # 1. Scegliamo un momento a caso nel passato
        pivot_date = random.choice(trainable_dates)

        # Conversione sicura in datetime
        if isinstance(pivot_date, str):
            pivot_dt = datetime.strptime(pivot_date, "%Y-%m-%d %H:%M:%S")
        else:
            pivot_dt = pivot_date

        # 2. Recuperiamo il contesto (Passato)
        # Usiamo get_trading_context_traning che hai già
        # Nota: forecast_config qui lo mettiamo vuoto o fittizio se non hai dati forecast storici
        context = db.get_trading_context_traning(
            base_currency=CURRENCY,
            history_config=TF_CONFIG,
            pivot_timestamp=pivot_dt,
            history_config_forecast=tf_config_forecast, # Se non hai forecast storici salvati
            forecast_forward_tf="1d",
            forecast_limit=1
        )

        # Verifica integrità dati
        if not context['candles'].get('1h') or len(context['candles']['1h']) < TF_CONFIG['1h']:
            continue # Salta se mancano dati in questo punto

        # 3. Recuperiamo il Futuro (per l'Oracolo)
        # Calcoliamo data limite futuro
        future_limit_dt = pivot_dt + timedelta(hours=LOOKAHEAD_STEPS)

        # Recuperiamo candele tra pivot e future_limit
        # Nota: Qui serve una query custom o riutilizziamo un metodo esistente con logica inversa
        # Per semplicità, facciamo una query diretta qui o aggiungi un metodo in DB
        # Simuliamo recupero:
        future_candles = db.get_candles_with_offset("currency", "1h", CURRENCY, 0)
        # (Attenzione: questo metodo prende le ULTIME. A noi servono quelle DOPO il pivot)
        # Per farla bene senza impazzire col DB ora:
        # Recuperiamo un blocco grande attorno alla data e filtriamo in Python
        # (Metodo grezzo ma efficace per prototipo)

        # FIX RAPIDO PER RECUPERO FUTURO:
        # Poiché non abbiamo un metodo "get_candles_between", usiamo get_candles_before_date
        # passandogli future_limit_dt e prendendo quelle fino a pivot_dt
        raw_future = db.get_candles_before_date("currency", "1h", CURRENCY, future_limit_dt, LOOKAHEAD_STEPS + 10)
        # Filtriamo solo quelle > pivot_dt
        future_segment = [c for c in raw_future if db.is_after(str(c['timestamp']), str(pivot_dt))]

        if len(future_segment) < 5:
            continue

        # 4. Eseguiamo step di training
        metrics = trainer.train_step(context, pair_info.get('pair_limits'), future_segment)

        if metrics:
            loss = metrics['loss']
            moving_avg_loss = 0.95 * moving_avg_loss + 0.05 * loss if i > 0 else loss

            if i % 10 == 0:
                side_str = ["BUY", "SELL", "HOLD"][metrics['target_side']]
                pred_str = ["BUY", "SELL", "HOLD"][metrics['pred_side']]
                print(f"Epoch {i}/{EPOCHS} | Loss: {loss:.4f} (Avg: {moving_avg_loss:.4f}) | Target: {side_str} vs Pred: {pred_str}")

    # Fine Training
    trainer.save_checkpoint("trm_model_v1.pth")
    db.close_connection()

if __name__ == "__main__":
    train_loop()
