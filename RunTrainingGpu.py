import sys
import os
import random
import torch
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
    TF_CONFIG = {"1d": 30, "4h": 50, "1h": 100}
    LOOKAHEAD_STEPS = 24  # 24 ore nel futuro per l'Oracolo
    EPOCHS = 300           # Numero di passaggi completi su tutte le coppie
    CACHE_LIMIT = 15000    # Quante candele scaricare per ogni coppia (per coprire storico + futuro)

    # --- SETUP ---
    print("--- INIZIALIZZAZIONE DB E PROVIDER ---")
    db = DatabaseManager()
    market_prov = MarketDataProvider()

    # 1. Recuperiamo le coppie
    all_pairs = market_prov.getAllPairs(quote_filter="EUR", leverage_only=True)

    if not all_pairs:
        print("❌ Nessuna coppia trovata nel DB.")
        return

    print(f"--- TROVATE {len(all_pairs)} COPPIE. ---")

    # =========================================================================
    # 2. PRE-CACHING DEI DATI (OTTIMIZZAZIONE PER NGROK/COLAB)
    # =========================================================================
    # Scarichiamo tutto subito per evitare la latenza di rete durante il training
    print("⏳ INIZIO SCARICAMENTO DATI IN RAM (Attendere prego, evita lag ngrok)...")

    data_cache = {} # Struttura: { "ETH": { "1h": [...], "4h": [...] } }

    for i, pair_data in enumerate(all_pairs):
        currency = pair_data['base']
        print(f"   [{i+1}/{len(all_pairs)}] Caching {currency}...", end="\r")

        pair_cache = {}
        has_error = False

        for tf in TF_CONFIG.keys():
            # Scarica massivamente le ultime N candele
            # Usiamo select_all per velocità, ordinando per timestamp
            try:
                # Nota: la query è costruita per prendere le ultime CACHE_LIMIT
                query_where = f"base='{currency}' AND timeframe='{tf}' ORDER BY timestamp DESC LIMIT {CACHE_LIMIT}"
                rows = db.select_all("currency", query_where)

                # Importante: Riordiniamo dal più vecchio al più recente per affettare le liste correttamente
                # Gestiamo sia stringhe che datetime per il sort
                rows.sort(key=lambda x: str(x['timestamp']))

                pair_cache[tf] = rows
            except Exception as e:
                print(f"\n❌ Errore scaricamento {currency} {tf}: {e}")
                has_error = True

        if not has_error:
            data_cache[currency] = pair_cache

    print(f"\n✅ CACHING COMPLETATO! Dati pronti in RAM.")
    # =========================================================================

    # Inizializza Vectorizer
    vec_config = VectorizerConfig(candle_history_config=TF_CONFIG)
    vectorizer = DataVectorizer(vec_config)

    # Inizializza Modello
    model = MultiTimeframeTRM(
        tf_configs=TF_CONFIG,
        input_size_per_candle=vectorizer.candle_dim,
        static_size=vectorizer.static_total_dim,
        hidden_dim=256
    )

    # --- SETUP GPU ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 TRAINING SU DISPOSITIVO: {device}")
    if device.type == 'cuda':
        print(f"   Scheda Video: {torch.cuda.get_device_name(0)}")

    model.to(device)

    # Inizializza Trainer
    trainer = TradingTrainer(model, db, vectorizer)

    # --- TRAINING LOOP ---
    moving_avg_loss = 0.0
    best_loss = float(1.97)
    global_step = 0

    for epoch in range(EPOCHS):
        print(f"\n=== EPOCH {epoch+1}/{EPOCHS} ===")

        # Mischiamo le coppie per variare il training
        random.shuffle(all_pairs)
        epoch_losses = []

        for pair_data in all_pairs:
            pair_name = pair_data['pair']
            currency = pair_data['base']

            # 1. Recupero dati dalla RAM (Super veloce)
            cached_pair_data = data_cache.get(currency)
            if not cached_pair_data:
                continue

            # Usiamo 1h come riferimento temporale principale
            candles_1h = cached_pair_data.get('1h', [])

            # Servono abbastanza dati per storico + futuro
            required_len = TF_CONFIG['1h'] + LOOKAHEAD_STEPS + 10
            if len(candles_1h) < required_len:
                continue

            # 2. Selezione Pivot Casuale
            # Deve esserci spazio prima (per lo storico) e dopo (per il futuro)
            min_idx = TF_CONFIG['1h'] + 2
            max_idx = len(candles_1h) - LOOKAHEAD_STEPS - 2

            if min_idx >= max_idx:
                continue

            pivot_idx = random.randint(min_idx, max_idx)
            pivot_candle = candles_1h[pivot_idx]

            # Otteniamo il timestamp del pivot come stringa per confronti uniformi
            pivot_ts_val = pivot_candle['timestamp']
            pivot_ts_str = str(pivot_ts_val)

            # 3. Costruzione Contesto (Slicing in Memoria)
            context = {
                "candles": {},
                "order": None,      # Per velocità training ignoriamo ordini aperti simulati per ora
                "forecast": [],     # Forecast vuoto o implementabile similmente
                "wallet_balance": 0.0
            }

            valid_context = True
            for tf, limit in TF_CONFIG.items():
                tf_data = cached_pair_data.get(tf, [])
                if not tf_data:
                    valid_context = False
                    break

                # Filtriamo: prendiamo le candele <= pivot_time
                # Essendo la lista ordinata, potremmo ottimizzare, ma la list comprehension in RAM è veloce
                # Cerchiamo le candele passate
                past_candles = [c for c in tf_data if str(c['timestamp']) <= pivot_ts_str]

                if len(past_candles) < limit:
                    valid_context = False
                    break

                # Prendiamo solo le ultime 'limit' necessarie
                context["candles"][tf] = past_candles[-limit:]

            if not valid_context:
                continue

            # 4. Costruzione Futuro (Oracolo) dalla RAM
            # Prendiamo le candele successive al pivot nella lista 1h
            # Slice: da pivot_idx + 1 fino a pivot_idx + 1 + LOOKAHEAD
            future_segment = candles_1h[pivot_idx+1 : pivot_idx+1+LOOKAHEAD_STEPS]

            if len(future_segment) < 5:
                continue

            # 5. Training Step
            # Passiamo i dati al trainer. Il trainer sposterà i tensori su GPU internamente.
            metrics = trainer.train_step(context, pair_data.get('pair_limits'), future_segment, global_step)

            if metrics:
                global_step += 1
                loss = metrics['loss']
                epoch_losses.append(loss)

                # Aggiornamento media mobile
                moving_avg_loss = 0.99 * moving_avg_loss + 0.01 * loss if global_step > 1 else loss

                # Log meno frequente per non intasare l'output
                if global_step % 20 == 0:
                    side_str = ["BUY", "SELL", "HOLD"][metrics['target_side']]
                    pred_str = ["BUY", "SELL", "HOLD"][metrics['pred_side']]
                    print(f"[Ep {epoch+1}][Step {global_step}] {pair_name} | Loss: {loss:.4f} (Avg: {moving_avg_loss:.4f}) | Target: {side_str} vs Pred: {pred_str}")

        # === FINE EPOCA ===
        if len(epoch_losses) > 0:
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            print(f"--- FINE EPOCA {epoch+1} | Media Loss: {avg_epoch_loss:.5f} ---")

            # Salvataggio Best Model
            if avg_epoch_loss < best_loss:
                print(f"🌟 NUOVO RECORD (Old: {best_loss:.5f} -> New: {avg_epoch_loss:.5f}) - Salvataggio...")
                best_loss = avg_epoch_loss
                trainer.save_checkpoint("trm_model_best.pth")
            else:
                print(f"--- Nessun miglioramento (Best: {best_loss:.5f}) ---")

        # Checkpoint regolare
        trainer.save_checkpoint("trm_model_v3.pth")

    # Chiusura
    trainer.save_checkpoint("trm_model_v3.pth")
    db.close_connection()
    print("--- TRAINING COMPLETATO ---")

if __name__ == "__main__":
    train_loop()
