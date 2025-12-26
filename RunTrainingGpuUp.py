import sys
import os
import random
import math
import torch
from datetime import datetime, timedelta

# --- AMD ROCm Windows Fix ---
# Fix per errore "rocrand_xorwow.h file not found" su Windows con GPU AMD
if os.name == 'nt':
    hip_path = os.environ.get('HIP_PATH')
    if hip_path:
        # Assicura che hiprtc trovi i file header in include/
        include_path = os.path.join(hip_path, 'include')
        current_cpath = os.environ.get('CPATH', '')
        if include_path not in current_cpath:
            os.environ['CPATH'] = f"{include_path};{current_cpath}"
            print(f"🔧 ROCm Windows Fix: Aggiunto {include_path} a CPATH per compilazione JIT.")
# ----------------------------

# Path Setup
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from db.DatabaseManager import DatabaseManager
from db.MarketDataProvider import MarketDataProvider
from trading.Vectorizer import DataVectorizer, VectorizerConfig
from trading.TrmAgent import MultiTimeframeTRM
from trading.TrainerUp import TradingTrainer
from utils.plotting import plot_training_losses

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


def simulate_pnl_from_preds(preds, current_price, future_segment, fee_rate=0.001):
    """
    Validazione rapida PnL (EUR) su un singolo sample.
    Assunzioni semplici ma coerenti:
      - side: 0 BUY, 1 SELL, 2 HOLD
      - entry: current_price
      - exit: hit-test su TP/SL usando high/low delle candele future (fallback: close ultima candela)
      - notional fisso: 100€ * qty_frac * leverage
      - fee_rate applicata 2 volte (entry+exit)
    """
    side = int(torch.argmax(preds["side"]).item())
    if side == 2:
        return 0.0

    entry = float(current_price)
    qty_frac = float(preds["qty"].clamp(0, 1).item())
    lev = float(preds["leverage"].clamp(1, 5).item())

    notional = 100.0 * qty_frac * lev
    if notional <= 0.0 or entry <= 0.0:
        return 0.0

    tp_mult = float(preds["tp_mult"].clamp(0, 10).item())
    sl_mult = float(preds["sl_mult"].clamp(0, 10).item())

    # Coerente con scaling del Trainer: tp_mult ~ pct/0.10, sl_mult ~ pct/0.05
    tp_pct = max(0.0, min(0.10 * tp_mult, 0.50))
    sl_pct = max(0.0, min(0.05 * sl_mult, 0.50))

    if side == 0:  # BUY
        tp = entry * (1.0 + tp_pct)
        sl = entry * (1.0 - sl_pct)
        exit_px = float(future_segment[-1]["close"])
        for c in future_segment:
            if float(c["low"]) <= sl:
                exit_px = sl
                break
            if float(c["high"]) >= tp:
                exit_px = tp
                break
        raw_ret = (exit_px - entry) / entry
    else:  # SELL
        tp = entry * (1.0 - tp_pct)
        sl = entry * (1.0 + sl_pct)
        exit_px = float(future_segment[-1]["close"])
        for c in future_segment:
            if float(c["high"]) >= sl:
                exit_px = sl
                break
            if float(c["low"]) <= tp:
                exit_px = tp
                break
        raw_ret = (entry - exit_px) / entry

    net_ret = raw_ret - 2.0 * fee_rate
    return float(notional * net_ret)

def train_loop():
    # --- CONFIGURAZIONE ---
    TF_CONFIG = {"1d": 30, "4h": 50, "1h": 100}
    # Configurazione Forecast
    TF_CONFIG_FORECAST = {"1d+1": 1, "1d+2": 1, "4h+1": 1, "4h+2": 1}
    FORECAST_FORWARD_TF = "1d" # Quanto in avanti guardiamo per selezionare il forecast

    LOOKAHEAD_STEPS = 24  # 24 ore nel futuro per l'Oracolo (Target)
    EPOCHS = 700
    CACHE_LIMIT = 300000    # Candele storiche
    FORECAST_CACHE_LIMIT = 100000 # Forecast limit

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
    best_penalized_score = float('-inf')  # Inizializa a -inf per accettare qualunque score
    global_step = 0
    loss_history = []

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

            split_idx = int(min_idx + 0.80 * (max_idx - min_idx))
            if split_idx <= min_idx:
                continue
            if split_idx >= max_idx:
                split_idx = max_idx - 1

            # TRAIN: scegli pivot nella parte iniziale (walk-forward)
            pivot_idx = random.randint(min_idx, split_idx)
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
            # IMPORTANTE: In training NON dobbiamo vedere forecast generati DOPO il pivot_dt.
            missing_forecast_for_this_pivot = False

            if cached_forecast_data:
                for tf_fc, limit_fc in TF_CONFIG_FORECAST.items():
                    fc_rows = cached_forecast_data.get(tf_fc, [])
                    if not fc_rows:
                        continue

                    # FILTRO CORRETTO: Solo forecast passati o presenti
                    valid_fc = [f for f in fc_rows if f['timestamp_dt'] <= pivot_dt]

                    if valid_fc:
                        # Prendiamo gli ultimi 'limit_fc' validi
                        candidate_fc = valid_fc[-limit_fc:]
                        last_candidate = candidate_fc[-1]

                        # STALENESS CHECK
                        time_diff = pivot_dt - last_candidate['timestamp_dt']

                        if time_diff > timedelta(hours=48):
                            missing_forecast_for_this_pivot = True
                            break

                        context["forecast"].extend(candidate_fc)
                    else:
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

                if global_step % 10 == 0:
                    loss_history.append({k: (v if not hasattr(v, 'item') else v.item()) for k, v in metrics.items() if k.startswith('loss')})
                    side_str = ["BUY", "SELL", "HOLD"][metrics['target_side']]
                    pred_str = ["BUY", "SELL", "HOLD"][metrics['pred_side']]
                    print(f"[Ep {epoch+1}][Step {global_step}] {pair_name} | Loss: {loss:.4f} (Avg: {moving_avg_loss:.4f}) | RL Loss: {metrics.get('loss_rl', 0):.4f} | T: {side_str} vs P: {pred_str}")


        # === FINE EPOCA ===
        if len(epoch_losses) > 0:
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            print(f"--- FINE EPOCA {epoch+1} | Media Loss: {avg_epoch_loss:.5f} ---")

            # === VALIDAZIONE WALK-FORWARD (PnL) ===
            model.eval()
            val_samples = 200
            val_pnls = []
            val_trades = []

            with torch.no_grad():
                for _ in range(val_samples):
                    pair_data_v = random.choice(all_pairs)
                    currency_v = pair_data_v["base"]

                    cached_pair_data_v = data_cache.get(currency_v)
                    cached_forecast_data_v = forecast_cache.get(currency_v, {})
                    if not cached_pair_data_v or not cached_forecast_data_v:
                        continue

                    candles_1h_v = cached_pair_data_v.get("1h", [])
                    required_len_v = TF_CONFIG["1h"] + LOOKAHEAD_STEPS + 10
                    if len(candles_1h_v) < required_len_v:
                        continue

                    min_idx_v = TF_CONFIG["1h"] + 2
                    max_idx_v = len(candles_1h_v) - LOOKAHEAD_STEPS - 2
                    if min_idx_v >= max_idx_v:
                        continue

                    split_idx_v = int(min_idx_v + 0.80 * (max_idx_v - min_idx_v))
                    if split_idx_v + 1 >= max_idx_v:
                        continue

                    # VAL: scegli pivot nella parte finale (forward)
                    pivot_idx_v = random.randint(split_idx_v + 1, max_idx_v)
                    pivot_candle_v = candles_1h_v[pivot_idx_v]
                    pivot_dt_v = pivot_candle_v["timestamp_dt"]

                    # Costruzione contesto come in training
                    context_v = {"candles": {}, "order": None, "forecast": [], "wallet_balance": 0.0}

                    valid_context_v = True
                    for tf, limit in TF_CONFIG.items():
                        tf_data_v = cached_pair_data_v.get(tf, [])
                        if not tf_data_v:
                            valid_context_v = False; break
                        past_candles_v = [c for c in tf_data_v if c["timestamp_dt"] <= pivot_dt_v]
                        if len(past_candles_v) < limit:
                            valid_context_v = False; break
                        context_v["candles"][tf] = past_candles_v[-limit:]

                    if not valid_context_v:
                        continue

                    # Forecast (stessa logica / freschezza)
                    missing_fc_v = False

                    for tf_fc, limit_fc in TF_CONFIG_FORECAST.items():
                        fc_rows_v = cached_forecast_data_v.get(tf_fc, [])
                        if not fc_rows_v:
                            continue

                        # FILTRO CORRETTO: <= pivot_dt
                        valid_fc_v = [f for f in fc_rows_v if f["timestamp_dt"] <= pivot_dt_v]

                        if valid_fc_v:
                            candidate_fc_v = valid_fc_v[-limit_fc:]
                            last_candidate_v = candidate_fc_v[-1]

                            time_diff_v = pivot_dt_v - last_candidate_v["timestamp_dt"]
                            if time_diff_v > timedelta(hours=48):
                                missing_fc_v = True
                                break
                            context_v["forecast"].extend(candidate_fc_v)
                        else:
                            missing_fc_v = True
                            break

                    if missing_fc_v or not context_v["forecast"]:
                        continue

                    future_segment_v = candles_1h_v[pivot_idx_v+1 : pivot_idx_v+1+LOOKAHEAD_STEPS]
                    if len(future_segment_v) < 5:
                        continue

                    pair_inputs_v = pair_data_v.get("pair_limits")
                    if not pair_inputs_v:
                        continue
                    pair_inputs_v = dict(pair_inputs_v)
                    pair_inputs_v["pair"] = pair_data_v.get("pair")
                    pair_inputs_v["base"] = pair_data_v.get("base")
                    pair_inputs_v["quote"] = pair_data_v.get("quote")
                    pair_inputs_v["kr_pair"] = pair_data_v.get("kr_pair")

                    inputs_v, ref_price_v = vectorizer.vectorize(
                        candles_db_data=context_v["candles"],
                        open_order=None,
                        forecast_db_data=context_v["forecast"],
                        pair_limits=pair_inputs_v,
                        wallet_balance=1000.0
                    )
                    inputs_v = {k: v.to(device) for k, v in inputs_v.items()}

                    brain_v = model.extract_features(inputs_v)
                    h_v = None
                    preds_v = None
                    for s in range(trainer.thinking_steps):
                        y_v, h_v = model.think(brain_v, h_v)
                        preds_v = model.get_heads_dict(y_v)

                    current_price_v = float(context_v["candles"]["1h"][-1]["close"])
                    pnl_v = simulate_pnl_from_preds(preds_v, current_price_v, future_segment_v, fee_rate=0.001)
                    val_pnls.append(pnl_v)

                    # Calcola trade flag
                    side_v = int(torch.argmax(preds_v["side"]).item())
                    val_trades.append(1 if side_v != 2 else 0)

            # === CALCOLO SCORE PENALIZZATO ===
            # Skip validation se non abbiamo campioni
            if len(val_pnls) == 0:
                print(f"⚠️  SKIP validazione (nessun campione)")
                model.train()
                continue

            avg_val_pnl = sum(val_pnls) / len(val_pnls)
            total_trades = sum(val_trades) if val_trades else 0
            trade_freq = total_trades / len(val_pnls)  # Normalizzato: 0-1

            # Iperparametri di penalizzazione (tunable)
            lambda_trade = 0.5   # Penalità PER FREQUENZA TRADING (non assoluto)

            # Score = PnL medio - penalità proporzionale alla frequenza trading
            penalized_score = avg_val_pnl - lambda_trade * trade_freq

            # Sanity check
            if not math.isfinite(penalized_score):
                print(f"⚠️  SKIP: score non finito (NaN/Inf), saltando salvataggio")
                model.train()
                continue

            print(f"📈 VAL PnL avg (EUR) = {avg_val_pnl:.4f} | Trades: {total_trades}/{len(val_pnls)} (freq={trade_freq*100:.1f}%)")
            print(f"📊 Score (penalizzato) = {penalized_score:.4f} = {avg_val_pnl:.4f} - {lambda_trade}*{trade_freq:.3f}")
            model.train()

            # Salva il miglior modello per score penalizzato
            if penalized_score > best_penalized_score + 1e-8:
                print(f"🌟 NUOVO BEST SCORE (Old: {best_penalized_score:.4f} -> New: {penalized_score:.4f}) - Salvataggio...")
                best_penalized_score = penalized_score
                trainer.save_checkpoint("model/trainerUpPnl.pth")
            else:
                trainer.save_checkpoint("model/trainerUpN.pth")
                print(f"--- Nessun miglioramento SCORE (Best: {best_penalized_score:.4f}) ---")

        db.close_connection()

        # Plotting finale
        plot_training_losses(loss_history)


if __name__ == "__main__":
    train_loop()
