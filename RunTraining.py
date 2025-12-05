import sys
import os
import random
from datetime import datetime, timedelta
import torch
# Path Setup
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from db.DatabaseManager import DatabaseManager
from db.MarketDataProvider import MarketDataProvider
from trading.Vectorizer import DataVectorizer, VectorizerConfig
from trading.TrmAgent import MultiTimeframeTRM
from trading.Trainer import TradingTrainer

def train_loop():
    # --- CONFIGURAZIONE ---
    TF_CONFIG = {"1d": 30, "4h": 50, "1h": 100}
    LOOKAHEAD_STEPS = 24 # 24 ore nel futuro per l'Oracolo

    # ORA EPOCHS SIGNIFICA: "Quante volte voglio passare su TUTTE le coppie?"
    # Se hai 50 coppie e metti 100 Epoche, farai 5.000 training step totali.
    # EPOCHS = 100
    EPOCHS = 50


    # --- SETUP ---
    db = DatabaseManager()
    market_prov = MarketDataProvider()

    # 1. Recuperiamo TUTTE le coppie disponibili (come in RunRefinedAgent)
    all_pairs = market_prov.getAllPairs(quote_filter="EUR", leverage_only=True)

    if not all_pairs:
        print("Nessuna coppia trovata nel DB.")
        return

    print(f"--- TROVATE {len(all_pairs)} COPPIE. INIZIO TRAINING... ---")

    # Inizializza Vectorizer (basta farlo una volta, la config è uguale per tutti)
    vec_config = VectorizerConfig(candle_history_config=TF_CONFIG)
    vectorizer = DataVectorizer(vec_config)

    # Inizializza Modello
    model = MultiTimeframeTRM(
        tf_configs=TF_CONFIG,
        input_size_per_candle=vectorizer.candle_dim,
        static_size=vectorizer.static_total_dim,
        hidden_dim=256
    )

    trainer = TradingTrainer(model, db, vectorizer)

    # --- TRAINING LOOP ANNIDATO ---
    moving_avg_loss = 0.0
    best_loss = float(2.5) # La miglior loss mai vista (inizia infinita)
    tf_config_forecast = {"1d+1": 1, "1d+2": 1, "4h+1": 1, "4h+2": 1}

    global_step = 0 # Contatore totale passi

    for epoch in range(EPOCHS):
        print(f"\n=== EPOCH {epoch+1}/{EPOCHS} ===")

        # Mischiamo le coppie a ogni epoca!
        # È fondamentale per non far imparare al modello sequenze fisse (prima BTC, poi ETH...)
        random.shuffle(all_pairs)
        epoch_losses = []  # Lista per tenere traccia delle loss di questa epoca
        for pair_data in all_pairs:
            pair_name = pair_data['pair']
            currency = pair_data['base']

            # --- A. Selezioniamo un momento a caso per QUESTA coppia ---
            # Query leggera solo timestamp
            candles = db.select_all("currency", f"base='{currency}' AND timeframe='1h' ORDER BY timestamp DESC LIMIT 3000")

            if len(candles) < 200:
                # print(f"Skipping {pair_name}: pochi dati")
                continue

            valid_dates = [c['timestamp'] for c in candles]
            # Tagliamo le date recenti per avere spazio per il futuro
            trainable_dates = valid_dates[LOOKAHEAD_STEPS:]

            if not trainable_dates:
                continue

            pivot_date = random.choice(trainable_dates)
            if isinstance(pivot_date, str):
                pivot_dt = datetime.strptime(pivot_date, "%Y-%m-%d %H:%M:%S")
            else:
                pivot_dt = pivot_date

            # --- B. Recupero Contesto (Passato) ---
            context = db.get_trading_context_traning(
                base_currency=currency,
                history_config=TF_CONFIG,
                pivot_timestamp=pivot_dt,
                history_config_forecast=tf_config_forecast,
                forecast_forward_tf="1d",
                forecast_limit=1
            )
            # ### NUOVO: SPOSTA IL MODELLO SU GPU SE DISPONIBILE
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # print(f"--- DEVICE SELEZIONATO: {device} ---")
            # if device.type == 'cuda':
            #     print(f"--- Scheda Video: {torch.cuda.get_device_name(0)} ---")

            model.to(device)
            if not context['candles'].get('1h') or len(context['candles']['1h']) < TF_CONFIG['1h']:
                continue

            # --- C. Recupero Futuro (Oracolo) ---
            future_limit_dt = pivot_dt + timedelta(hours=LOOKAHEAD_STEPS)

            # Nota: prendiamo un buffer più ampio per essere sicuri di coprire il range
            raw_future = db.get_candles_before_date("currency", "1h", currency, future_limit_dt, LOOKAHEAD_STEPS + 10)
            # Filtriamo: Vogliamo candele DOPO il pivot
            future_segment = [c for c in raw_future if db.is_after(str(c['timestamp']), str(pivot_dt))]

            if len(future_segment) < 5:
                continue

            # --- D. Training Step ---
            metrics = trainer.train_step(context, pair_data.get('pair_limits'), future_segment, global_step)

            if metrics:
                global_step += 1
                loss = metrics['loss']
                epoch_losses.append(loss) # Aggiungi alla lista
                # Media mobile esponenziale per smussare il grafico della loss
                moving_avg_loss = 0.99 * moving_avg_loss + 0.01 * loss if global_step > 1 else loss

                # Logghiamo ogni tanto (es. ogni 10 coppie processate)
                if global_step % 10 == 0:
                    side_str = ["BUY", "SELL", "HOLD"][metrics['target_side']]
                    pred_str = ["BUY", "SELL", "HOLD"][metrics['pred_side']]
                    print(f"[Ep {epoch+1}][Step {global_step}] {pair_name} | Loss: {loss:.4f} (Avg: {moving_avg_loss:.4f}) | T: {side_str} vs P: {pred_str}")

        # Fine epoca: calcoliamo e mostriamo la loss media
        # === FINE EPOCA: VALUTAZIONE ===
        if len(epoch_losses) > 0:
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            print(f"--- FINE EPOCA {epoch+1} | Media Loss: {avg_epoch_loss:.5f} ---")

            # Salvataggio "Best Model"
            if avg_epoch_loss < best_loss:
                print(f"!!! NUOVO RECORD !!! (Old: {best_loss:.5f} -> New: {avg_epoch_loss:.5f})")
                best_loss = avg_epoch_loss
                trainer.save_checkpoint("trm_model_best.pth")
            else:
                print(f"--- Nessun miglioramento (Best: {best_loss:.5f}) ---")
        # Salva checkpoint alla fine di ogni epoca (importante!)
        trainer.save_checkpoint("trm_model_v3.pth")

    # Fine Training
    trainer.save_checkpoint("trm_model_v3.pth")
    db.close_connection()
    print("--- TRAINING COMPLETATO ---")

if __name__ == "__main__":
    train_loop()
