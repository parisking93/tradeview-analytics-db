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


def simulate_pnl_from_preds(preds, current_price, future_segment, fee_rate=0.001, verbose=False):
    """
    Validazione rapida PnL (EUR) su un singolo sample.
    Ritorna (pnl, is_trade) dove:
      - pnl: profitto/perdita in EUR
      - is_trade: 1 se è stato aperto un trade, 0 se HOLD

    Assunzioni semplici ma coerenti:
      - side: 0 BUY, 1 SELL, 2 HOLD
      - entry: current_price
      - exit: hit-test su TP/SL usando high/low delle candele future (fallback: close ultima candela)
      - notional fisso: 100€ * qty_frac * leverage
      - fee_rate applicata 2 volte (entry+exit)
    """
    side = int(torch.argmax(preds["side"]).item())
    if side == 2:
        return 0.0, 0  # HOLD: no trade

    # Determine ordertype and price offset (fill-aware)
    ordertype = int(torch.argmax(preds.get("ordertype")).item()) if preds.get("ordertype") is not None else 0
    # price_offset normalized in [-1,1] -> scale to +/-5%
    try:
        px_off = float(preds.get("price_offset").clamp(-1.0, 1.0).item()) * 0.05
    except Exception:
        px_off = 0.0

    entry = float(current_price)
    qty_frac = float(preds["qty"].clamp(0, 1).item())
    lev = float(preds["leverage"].clamp(1, 5).item())

    notional = 100.0 * qty_frac * lev
    if notional <= 0.0 or entry <= 0.0:
        return 0.0, 0

    tp_mult = float(preds["tp_mult"].clamp(0, 10).item())
    sl_mult = float(preds["sl_mult"].clamp(0, 10).item())

    # Coerente con Decoder: tp_mult * 0.10, sl_mult * 0.05
    MIN_TP_PCT = 0.004
    MIN_SL_PCT = 0.003
    MAX_TP_PCT = 0.10
    MAX_SL_PCT = 0.05

    tp_pct = max(MIN_TP_PCT, min(tp_mult * MAX_TP_PCT, 0.50))
    sl_pct = max(MIN_SL_PCT, min(sl_mult * MAX_SL_PCT, 0.50))

    # If LIMIT, compute entry from offset and verify fill on future_segment
    filled = True
    px_off_enforced = px_off  # Track whether px_off was clamped
    if ordertype == 0:  # LIMIT
        # Enforce directional and distance bounds for realistic LIMIT orders
        MIN_LIMIT_DIST = 0.002  # 0.2% minimum distance from current price
        MAX_LIMIT_DIST = 0.05   # 5% maximum distance from current price

        if side == 0:  # BUY
            # For BUY: limit MUST be below current (px_off < 0)
            # Clamp to [-MAX, -MIN]
            px_off_enforced = max(-MAX_LIMIT_DIST, min(px_off, -MIN_LIMIT_DIST))
            entry = float(current_price) * (1.0 + px_off_enforced)

            # If order became "marketable" (entry >= current), treat as MARKET not LIMIT
            if entry >= float(current_price):
                ordertype = 1  # Convert to MARKET
                entry = float(current_price)
                filled = True
            else:
                # Standard LIMIT fill check: any low <= entry
                filled = any(float(c["low"]) <= entry for c in future_segment)

        else:  # SELL
            # For SELL: limit MUST be above current (px_off > 0)
            # Clamp to [+MIN, +MAX]
            px_off_enforced = max(MIN_LIMIT_DIST, min(px_off, MAX_LIMIT_DIST))
            entry = float(current_price) * (1.0 + px_off_enforced)

            # If order became "marketable" (entry <= current), treat as MARKET not LIMIT
            if entry <= float(current_price):
                ordertype = 1  # Convert to MARKET
                entry = float(current_price)
                filled = True
            else:
                # Standard LIMIT fill check: any high >= entry
                filled = any(float(c["high"]) >= entry for c in future_segment)

        if not filled:
            # LIMIT not touched -> no trade executed in validation
            if verbose:
                side_str = ["BUY", "SELL", "HOLD"][side]
                ord_str = "LIMIT"
                # print(f"[VAL_DBG] side:{side_str} ord:{ord_str} px_off:{px_off:.6f} px_enforced:{px_off_enforced:.6f} ref:{current_price:.5f} limit:{entry:.5f} filled:False")
            return 0.0, 0

    # Compute TP/SL relative to (possibly adjusted) entry
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

    if verbose:
        side_str = ["BUY", "SELL", "HOLD"][side]
        ord_str = "MARKET" if ordertype == 1 else "LIMIT"
        # print(f"[VAL_DBG] side:{side_str} ord:{ord_str} px_off:{px_off:.6f} px_enforced:{px_off_enforced:.6f} ref:{current_price:.5f} entry:{entry:.5f} filled:{filled}")

    return float(notional * net_ret), 1  # 1 = trade effettuato

def train_loop():
    # --- CONFIGURAZIONE ---
    TF_CONFIG = {"1h": 30, "15m": 50, "5m": 100}

    # Configurazione Forecast
    TF_CONFIG_FORECAST = {"1h+1": 1, "1h+2": 1, "15m+1": 1, "15m+2": 1}
    FORECAST_FORWARD_TF = "1d" # Quanto in avanti guardiamo per selezionare il forecast

    LOOKAHEAD_STEPS = 180  # 24 ore nel futuro per l'Oracolo (Target)
    EPOCHS = 700
    CACHE_LIMIT = 500000    # Candele storiche
    FORECAST_CACHE_LIMIT = 500000 # Forecast limit

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
        # --- FIX: Disable MIOpen if JIT fails ---
        if os.name == 'nt':
            print("🔧 Windows ROCm: Disabilitazione MIOpen (backend cudnn) per evitare errori JIT 'redefinition of forward'.")
            torch.backends.cudnn.enabled = False
        # ----------------------------------------

    model.to(device)
    trainer = TradingTrainer(model, db, vectorizer)

    # --- TRAINING LOOP ---
    moving_avg_loss = 0.0
    best_penalized_score = float('-inf')  # Inizializa a -inf per accettare qualunque score
    global_step = 0

    for epoch in range(EPOCHS):
        print(f"\n=== EPOCH {epoch+1}/{EPOCHS} ===")
        simulated_wallet = 0
        if random.random() < 0.1:
            simulated_wallet = random.uniform(0.1, 15.0)
        else:
            simulated_wallet = random.uniform(400.0, 1000.0)

        random.shuffle(all_pairs)
        epoch_losses = []

        for pair_data in all_pairs:
            pair_name = pair_data['pair']
            currency = pair_data['base']

            cached_pair_data = data_cache.get(currency)
            cached_forecast_data = forecast_cache.get(currency, {})

            if not cached_pair_data:
                continue

            candles_1h = cached_pair_data.get('5m', [])
            required_len = TF_CONFIG['5m'] + LOOKAHEAD_STEPS + 10
            if len(candles_1h) < required_len:
                continue

            # Selezione Pivot Casuale
            min_idx = TF_CONFIG['5m'] + 2
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
            # In live (runGoogle), alle 12:00 non abbiamo forecast delle 13:00.
            # Quindi filter condition: fc_timestamp <= pivot_dt.

            missing_forecast_for_this_pivot = False

            if cached_forecast_data:
                for tf_fc, limit_fc in TF_CONFIG_FORECAST.items():
                    fc_rows = cached_forecast_data.get(tf_fc, [])
                    if not fc_rows:
                        continue

                    # FILTRO CORRETTO: Solo forecast passati o presenti
                    # find efficiently? fc_rows is sorted ASC.
                    # Linear scan backwards or filter. Given len ~100k, filter is ok but maybe slow.
                    # Optimization: fc_rows è ordinato per data. Binary search o bisect sarebbe top.
                    # Per ora filter classico ma corretto.
                    valid_fc = [f for f in fc_rows if f['timestamp_dt'] <= pivot_dt]

                    if valid_fc:
                        # Prendiamo gli ultimi 'limit_fc' validi (i più recenti rispetto al pivot)
                        candidate_fc = valid_fc[-limit_fc:]
                        last_candidate = candidate_fc[-1]

                        # STALENESS CHECK
                        # Se il forecast "più recente" che ho è comunque vecchio di 48 ore, c'è un buco nei dati.
                        time_diff = pivot_dt - last_candidate['timestamp_dt']

                        if time_diff > timedelta(hours=48):
                            missing_forecast_for_this_pivot = True
                            break

                        context["forecast"].extend(candidate_fc)
                    else:
                        # Non ho nessun forecast nel passato per questo pivot
                        missing_forecast_for_this_pivot = True
                        break
            else:
                missing_forecast_for_this_pivot = True

            # Se i forecast sono troppo vecchi o mancanti per questo specifico pivot, saltiamo il training step
            if missing_forecast_for_this_pivot:
                continue

            # 5. Costruzione Futuro
            future_segment = candles_1h[pivot_idx+1 : pivot_idx+1+LOOKAHEAD_STEPS]
            if len(future_segment) < 5:
                continue

            pair_inputs = pair_data.get('pair_limits')
            pair_inputs['pair'] = pair_data.get('pair'); pair_inputs['base'] = pair_data.get('base'); pair_inputs['quote'] = pair_data.get('quote'); pair_inputs['kr_pair'] = pair_data.get('kr_pair')

            # 6. Training Step
            metrics = trainer.train_step(context, pair_inputs, future_segment, global_step, True, simulated_wallet)

            if metrics:
                simulated_wallet = metrics['simulated_wallet']
                global_step += 1
                loss = metrics['loss']
                epoch_losses.append(loss)
                moving_avg_loss = 0.99 * moving_avg_loss + 0.01 * loss if global_step > 1 else loss

                if global_step % 10 == 0:
                    side_str = ["BUY", "SELL", "HOLD"][metrics['target_side']]
                    pred_str = ["BUY", "SELL", "HOLD"][metrics['pred_side']]
                    # Loggare le loss per head
                    print(f"[Ep {epoch+1}][Step {global_step}] {pair_name} | Loss: {loss:.4f} (Avg: {moving_avg_loss:.4f}) | RL Loss: {metrics.get('loss_rl', 0):.4f} | T: {side_str} vs P: {pred_str}")
                    print(f"  ├─ side:{metrics.get('loss_side', 0):.4f} | type:{metrics.get('loss_type', 0):.4f} | qty:{metrics.get('loss_qty', 0):.4f} | px:{metrics.get('loss_px', 0):.4f}")
                    print(f"  └─ tp:{metrics.get('loss_tp', 0):.4f} | sl:{metrics.get('loss_sl', 0):.4f} | lev:{metrics.get('loss_lev', 0):.4f} | halt:{metrics.get('loss_halt', 0):.4f}")


        # === FINE EPOCA ===
        if len(epoch_losses) > 0:
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            print(f"--- FINE EPOCA {epoch+1} | Media Loss: {avg_epoch_loss:.5f} ---")

            # --- SCHEDULER STEP (per-epoch, non per-sample!) ---
            trainer.scheduler.step(avg_epoch_loss)
            current_lr = trainer.optimizer.param_groups[0]['lr']
            print(f"📉 LR attuale: {current_lr:.2e}")

            # === VALIDAZIONE WALK-FORWARD (PnL) ===
            model.eval()
            val_samples = 200
            val_pnls = []
            val_trades = []  # Traccia numero di trade per campione

            with torch.no_grad():
                for _ in range(val_samples):
                    pair_data_v = random.choice(all_pairs)
                    currency_v = pair_data_v["base"]

                    cached_pair_data_v = data_cache.get(currency_v)
                    cached_forecast_data_v = forecast_cache.get(currency_v, {})
                    if not cached_pair_data_v or not cached_forecast_data_v:
                        continue

                    candles_1h_v = cached_pair_data_v.get("5m", [])
                    required_len_v = TF_CONFIG["5m"] + LOOKAHEAD_STEPS + 10
                    if len(candles_1h_v) < required_len_v:
                        continue

                    min_idx_v = TF_CONFIG["5m"] + 2
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
                    forecast_upper_dt_v = add_timeframe_py(pivot_dt_v, FORECAST_FORWARD_TF)
                    missing_fc_v = False

                    for tf_fc, limit_fc in TF_CONFIG_FORECAST.items():
                        fc_rows_v = cached_forecast_data_v.get(tf_fc, [])
                        if not fc_rows_v:
                            continue

                        valid_fc_v = [f for f in fc_rows_v if f["timestamp_dt"] < forecast_upper_dt_v]
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
                    pnl_v, is_trade_v = simulate_pnl_from_preds(preds_v, current_price_v, future_segment_v, fee_rate=0.001, verbose=True)
                    val_pnls.append(pnl_v)
                    val_trades.append(is_trade_v)

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
            lambda_trade = 0.7   # Penalità PER FREQUENZA TRADING (non assoluto)

            # Score = PnL medio - penalità proporzionale alla frequenza trading
            # Se fai il 100% di trade (1.0), sottrai lambda_trade * 1.0 = 0.5 dal PnL
            # Se fai il 50% di trade (0.5), sottrai lambda_trade * 0.5 = 0.25 dal PnL
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
            if penalized_score > best_penalized_score + 1e-8:  # epsilon per evitare flip su valori molto vicini
                print(f"🌟 NUOVO BEST SCORE (Old: {best_penalized_score:.4f} -> New: {penalized_score:.4f}) - Salvataggio...")
                best_penalized_score = penalized_score
                trainer.save_checkpoint("model/trainerBest.pth")
            else:
                trainer.save_checkpoint("model/trainerLast.pth")
                print(f"--- Nessun miglioramento SCORE (Best: {best_penalized_score:.4f}) ---")

        db.close_connection()

if __name__ == "__main__":
    train_loop()
