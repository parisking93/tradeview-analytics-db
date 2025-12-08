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

def add_timeframe_py(dt_obj, timeframe_str):
    """
    Replica la logica di add_timeframe del DB ma in Python puro per velocità.
    """
    mapping = {
        "1m": timedelta(minutes=1),
        "5m": timedelta(minutes=5),
        "15m": timedelta(minutes=15),
        "1h": timedelta(hours=1),
        "4h": timedelta(hours=4),
        "1d": timedelta(days=1),
    }
    delta = mapping.get(timeframe_str)
    if delta:
        return dt_obj + delta
    return dt_obj # Fallback

def get_timedelta_from_tf(timeframe_str):
    mapping = {
        "1m": timedelta(minutes=1),
        "5m": timedelta(minutes=5),
        "15m": timedelta(minutes=15),
        "1h": timedelta(hours=1),
        "4h": timedelta(hours=4),
        "1d": timedelta(days=1),
    }
    # Rimuovi eventuali suffissi tipo "+1", "+2" per ottenere la base temporale
    base_tf = timeframe_str.split('+')[0]
    return mapping.get(base_tf, timedelta(hours=1))

def to_datetime(ts):
    """Utility per convertire timestamp in datetime in modo sicuro"""
    if isinstance(ts, datetime):
        return ts
    try:
        # Tenta formato standard DB
        return datetime.strptime(str(ts), "%Y-%m-%d %H:%M:%S")
    except:
        return None

def train_loop():
    # --- CONFIGURAZIONE ---
    TF_CONFIG = {"1d": 30, "4h": 50, "1h": 100}

    # Configurazione Forecast
    TF_CONFIG_FORECAST = {"1d+1": 1, "1d+2": 1, "4h+1": 1, "4h+2": 1}
    FORECAST_FORWARD_TF = "1d" # Quanto in avanti guardiamo per selezionare il forecast

    LOOKAHEAD_STEPS = 24  # 24 ore nel futuro per l'Oracolo (Target)
    EPOCHS = 400
    CACHE_LIMIT = 200000    # Candele storiche
    FORECAST_CACHE_LIMIT = 40000 # Forecast limit

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
    # 2. PRE-CACHING DEI DATI
    # =========================================================================
    print("⏳ INIZIO SCARICAMENTO DATI IN RAM (Candele + Forecast)...")

    data_cache = {}
    forecast_cache = {}

    for i, pair_data in enumerate(all_pairs):
        currency = pair_data['base']
        print(f"   [{i+1}/{len(all_pairs)}] Caching {currency}...", end="\r")

        # --- A. Caching Candele Storiche (Currency) ---
        pair_cache = {}
        has_error = False
        try:
            for tf in TF_CONFIG.keys():
                query_where = f"base='{currency}' AND timeframe='{tf}' ORDER BY timestamp DESC LIMIT {CACHE_LIMIT}"
                rows = db.select_all("currency", query_where)
                # Convertiamo subito timestamp in datetime objects per evitare cast continui dopo
                for r in rows:
                    r['timestamp_dt'] = to_datetime(r['timestamp'])

                rows.sort(key=lambda x: x['timestamp_dt']) # Sort ASC
                pair_cache[tf] = rows
        except Exception as e:
            print(f"\n❌ Errore scaricamento Candele {currency}: {e}")
            has_error = True

        if not has_error:
            data_cache[currency] = pair_cache

        # --- B. Caching Forecast ---
        pair_fc_cache = {}
        try:
            for tf_fc in TF_CONFIG_FORECAST.keys():
                query_where_fc = f"base='{currency}' AND timeframe='{tf_fc}' ORDER BY timestamp DESC LIMIT {FORECAST_CACHE_LIMIT}"
                rows_fc = db.select_all("forecast", query_where_fc)

                for r in rows_fc:
                    r['timestamp_dt'] = to_datetime(r['timestamp'])

                rows_fc.sort(key=lambda x: x['timestamp_dt']) # Sort ASC
                pair_fc_cache[tf_fc] = rows_fc
        except Exception as e:
            pass

        if pair_fc_cache:
            forecast_cache[currency] = pair_fc_cache

    print(f"\n✅ CACHING COMPLETATO! Dati pronti in RAM.")
    # =========================================================================

    vec_config = VectorizerConfig(candle_history_config=TF_CONFIG)
    vectorizer = DataVectorizer(vec_config)

    model = MultiTimeframeTRM(
        tf_configs=TF_CONFIG,
        input_size_per_candle=vectorizer.candle_dim,
        static_size=vectorizer.static_total_dim,
        hidden_dim=512
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 TRAINING SU DISPOSITIVO: {device}")
    if device.type == 'cuda':
        print(f"   Scheda Video: {torch.cuda.get_device_name(0)}")

    model.to(device)
    trainer = TradingTrainer(model, db, vectorizer)

    # --- TRAINING LOOP ---
    moving_avg_loss = 0.0
    best_loss = float(3.0)
    global_step = 0

    for epoch in range(EPOCHS):
        print(f"\n=== EPOCH {epoch+1}/{EPOCHS} ===")

        random.shuffle(all_pairs)
        epoch_losses = []

        for pair_data in all_pairs:
            pair_name = pair_data['pair']
            currency = pair_data['base']

            cached_pair_data = data_cache.get(currency)
            cached_forecast_data = forecast_cache.get(currency, {})

            if not cached_pair_data:
                continue

            candles_1h = cached_pair_data.get('1h', [])
            required_len = TF_CONFIG['1h'] + LOOKAHEAD_STEPS + 10
            if len(candles_1h) < required_len:
                continue

            # Selezione Pivot Casuale
            min_idx = TF_CONFIG['1h'] + 2
            max_idx = len(candles_1h) - LOOKAHEAD_STEPS - 2

            if min_idx >= max_idx:
                continue

            pivot_idx = random.randint(min_idx, max_idx)
            pivot_candle = candles_1h[pivot_idx]

            # Usiamo il datetime pre-calcolato
            pivot_dt = pivot_candle['timestamp_dt']
            pivot_ts_str = str(pivot_candle['timestamp'])

            # Costruzione Contesto
            context = {
                "candles": {},
                "order": None,
                "forecast": [],
                "wallet_balance": 0.0
            }

            valid_context = True
            for tf, limit in TF_CONFIG.items():
                tf_data = cached_pair_data.get(tf, [])
                if not tf_data:
                    valid_context = False; break

                # Slicing ottimizzato in memoria
                # Troviamo l'indice del pivot o subito prima (binary search sarebbe meglio, ma linear scan su 15k è ok in ram)
                # Qui facciamo filtraggio classico per sicurezza
                past_candles = [c for c in tf_data if c['timestamp_dt'] <= pivot_dt]

                if len(past_candles) < limit:
                    valid_context = False; break

                context["candles"][tf] = past_candles[-limit:]

            if not valid_context:
                continue

            # 4. Costruzione FORECAST
            # Calcolo limite superiore: Pivot + Forward TF
            forecast_upper_dt = add_timeframe_py(pivot_dt, FORECAST_FORWARD_TF)

            missing_forecast_for_this_pivot = False

            if cached_forecast_data:
                for tf_fc, limit_fc in TF_CONFIG_FORECAST.items():
                    fc_rows = cached_forecast_data.get(tf_fc, [])
                    if not fc_rows:
                        # Se manca una tipologia intera di forecast per questa moneta, è ok o skip?
                        # Per ora continuiamo provando a prenderne altri, o break se rigidi.
                        continue

                    # Logica: Cerchiamo forecast con timestamp < forecast_upper_dt
                    valid_fc = [f for f in fc_rows if f['timestamp_dt'] < forecast_upper_dt]

                    if valid_fc:
                        # --- CONTROLLO FRESCHEZZA (STALENESS CHECK) ---
                        # Prendiamo l'ultimo disponibile
                        candidate_fc = valid_fc[-limit_fc:]
                        last_candidate = candidate_fc[-1]

                        # Calcoliamo quanto è vecchio questo forecast rispetto al pivot
                        # Se il pivot è oggi, e il forecast è di 3 giorni fa, non va bene.
                        # Tolleranza: 2 volte la durata del timeframe del forecast (es. 8 ore per 4h)
                        # O più semplicemente: se dista più di 48 ore dal pivot, è troppo vecchio.
                        time_diff = pivot_dt - last_candidate['timestamp_dt']

                        # Se time_diff è negativo, il forecast è nel futuro rispetto al pivot (non dovrebbe accadere col filtro < upper, ma ok)
                        # Se time_diff è positivo, il forecast è nel passato.

                        # Esempio: Pivot 17 Nov. Forecast 02 Nov. Diff = 15 giorni. -> STALE.
                        # Esempio: Pivot 17 Nov. Forecast 16 Nov. Diff = 1 giorno. -> OK.

                        if time_diff > timedelta(hours=48):
                            # Dato troppo vecchio, probabilmente il download limitato non copre questa data
                            # o il forecast non è stato generato.
                            missing_forecast_for_this_pivot = True
                            break # Interrompiamo questo pivot, inutile trainare su dati vecchi

                        context["forecast"].extend(candidate_fc)
                    else:
                        # Nessun forecast trovato prima della data target
                        missing_forecast_for_this_pivot = True
                        break
            else:
                missing_forecast_for_this_pivot = True

            # Se i forecast sono troppo vecchi o mancanti per questo specifico pivot, saltiamo il training step
            # Questo evita di addestrare il modello con "buco" nei dati input
            if missing_forecast_for_this_pivot:
                # print(f"Skip {pair_name} at {pivot_dt}: Forecast mancanti o vecchi.")
                continue

            # 5. Costruzione Futuro
            future_segment = candles_1h[pivot_idx+1 : pivot_idx+1+LOOKAHEAD_STEPS]
            if len(future_segment) < 5:
                continue

            pair_inputs = pair_data.get('pair_limits')
            pair_inputs['pair'] = pair_data.get('pair'); pair_inputs['base'] = pair_data.get('base'); pair_inputs['quote'] = pair_data.get('quote'); pair_inputs['kr_pair'] = pair_data.get('kr_pair')

            # 6. Training Step
            metrics = trainer.train_step(context, pair_inputs, future_segment, global_step)

            if metrics:
                global_step += 1
                loss = metrics['loss']
                epoch_losses.append(loss)
                moving_avg_loss = 0.99 * moving_avg_loss + 0.01 * loss if global_step > 1 else loss

                if global_step % 20 == 0:
                    side_str = ["BUY", "SELL", "HOLD"][metrics['target_side']]
                    pred_str = ["BUY", "SELL", "HOLD"][metrics['pred_side']]
                    print(f"[Ep {epoch+1}][Step {global_step}] {pair_name} | Loss: {loss:.4f} (Avg: {moving_avg_loss:.4f}) | T: {side_str} vs P: {pred_str}")

        # === FINE EPOCA ===
        if len(epoch_losses) > 0:
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            print(f"--- FINE EPOCA {epoch+1} | Media Loss: {avg_epoch_loss:.5f} ---")

            if avg_epoch_loss < best_loss:
                print(f"🌟 NUOVO RECORD (Old: {best_loss:.5f} -> New: {avg_epoch_loss:.5f}) - Salvataggio...")
                best_loss = avg_epoch_loss
                trainer.save_checkpoint("trm_model_best.pth")
            else:
                print(f"--- Nessun miglioramento (Best: {best_loss:.5f}) ---")

        trainer.save_checkpoint("trm_model_v3.pth")

    trainer.save_checkpoint("trm_model_v3.pth")
    db.close_connection()
    print("--- TRAINING COMPLETATO ---")

if __name__ == "__main__":
    train_loop()
