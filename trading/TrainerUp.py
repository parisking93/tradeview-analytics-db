import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from datetime import timedelta, datetime

class TradingTrainer:
    def __init__(self, model, db_manager, vectorizer, learning_rate=1e-4):
        self.model = model
        self.db = db_manager
        self.vectorizer = vectorizer

        # --- HALT tuning ---
        self.halt_gamma = 2.0          # focal gamma
        self.halt_alpha_pos = 0.75     # peso quando target_halt ~ 1 (HOLD)
        self.halt_alpha_neg = 0.25     # peso quando target_halt ~ 0 (BUY/SELL)
        self.halt_loss_weight = 0.35   # quanto pesa halt nella loss totale (prova 0.25–0.50)


        # Carica pesi (Best effort)
        try:
        #     self.model.load_state_dict(torch.load("trm_model_v2.pth"), strict=False)
            # self.model.load_state_dict(torch.load("trm_model_best.pth"), strict=False)
            print("--- Pesi 'Best Model' caricati ---")
        except:
            print("--- Nessun peso precedente, start fresh ---")

        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)

        # --- SCHEDULER ---
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=500
        )
        self.accumulation_steps = 4

        # --- MULTISTEP THINKING ---
        # Numero di step di "pensiero" durante il training (unroll della GRUCell del cervello)
        self.thinking_steps = 7  # puoi portarlo a 6 se vuoi avvicinarti al runGoogle

        # --- LOSS FUNCTIONS SEPARATE ---

        # 1. Loss per il SIDE (3 classi: Buy, Sell, Hold)
        # Qui USIAMO i pesi per bilanciare l'Hold
        self.weights_side = torch.tensor([2.5, 2.5, 1.0])
        self.loss_ce_side = nn.CrossEntropyLoss(weight=self.weights_side)

        # 2. Loss per ORDER TYPE (2 classi: Limit, Market)
        # Qui NON usiamo pesi (o standard), perche Limit e Market sono bilanciati
        self.loss_ce_type = nn.CrossEntropyLoss()

        self.loss_mse = nn.MSELoss()
        self.loss_bce = nn.BCELoss()

    def generate_fake_order(self, context, pair_limits):
        """
        Genera un ordine finto basato su candele di 1-2 giorni fa.
        Popola l'oggetto ordine come se venisse dal DB.
        """
        # Se c'è già un ordine reale (o finto generato in precedenza), usciamo
        if context.get('order') is not None:
            return None

        # Parametro di casualità: 40% di probabilità di avere un ordine aperto
        if random.random() > 0.2:
            return None

        # Recuperiamo le candele 1h
        candles = context['candles'].get('1h', [])
        if not candles or len(candles) < 10:
            return None

        current_candle = candles[-1]

        # Gestione sicura del datetime (supporta sia oggetti datetime che stringhe)
        ts_val = current_candle.get('timestamp_dt') # Se pre-calcolato da RunTrainingGpu
        if ts_val is None:
            # Fallback se non c'è timestamp_dt
            raw_ts = current_candle['timestamp']
            if isinstance(raw_ts, str):
                ts_val = datetime.strptime(raw_ts, "%Y-%m-%d %H:%M:%S")
            else:
                ts_val = raw_ts

        current_price = float(current_candle['close'])

        # --- LOGICA TEMPORALE ---
        # Cerchiamo una candela tra 24h e 48h fa
        target_start = ts_val - timedelta(hours=48)
        target_end = ts_val - timedelta(hours=24)

        # Filtriamo le candele candidate
        candidates = []
        for c in candles:
            # Parsing locale se necessario
            c_ts = c.get('timestamp_dt')
            if c_ts is None:
                raw = c['timestamp']
                c_ts = datetime.strptime(raw, "%Y-%m-%d %H:%M:%S") if isinstance(raw, str) else raw

            if target_start <= c_ts <= target_end:
                candidates.append(c)

        if not candidates:
            return None

        # Scelta casuale della candela di entry
        entry_candle = random.choice(candidates)
        price_entry = float(entry_candle['close'])

        randomC = random.choice([0,1,1])
        is_buy = False
        if randomC == 0:
            is_buy = random.choice([True, False])
        else:
            if current_price <= price_entry:
                is_buy = False
            else:
                is_buy = True
        # --- CREAZIONE ORDINE ---
        # Decidiamo casualmente Buy o Sell

        subtype = 'buy' if is_buy else 'sell'


        # type
        type = "position"
        lev = 1
        if is_buy == False:
            type = "position_margin"
            lev = random.choice([2,2,3,4,5])
        else:
            lev = random.choice([1,1,1,2,3,4,5])
            if lev != 1:
                type = "position_margin"

        # Simuliamo una size in Euro (es. tra 20 e 200 Euro)
        simulated_margin_eur = random.uniform(20.0, 200.0)
        total_position_value_eur = simulated_margin_eur * lev
        qty = total_position_value_eur / price_entry

        # Simuliamo TP e SL (es. +/- 2% e 5%)
        # Nota: Li popoliamo come float, il DB li ha come numeri
        tp_pct = random.uniform(0.02, 0.05)
        sl_pct = random.uniform(0.01, 0.03)

        if is_buy:
            take_profit = price_entry * (1 + tp_pct)
            stop_loss = price_entry * (1 - sl_pct)
            # PnL Buy: (Prezzo Attuale - Prezzo Entry) * Qty
            pnl = (current_price - price_entry) * qty
        else:
            take_profit = price_entry * (1 - tp_pct)
            stop_loss = price_entry * (1 + sl_pct)
            # PnL Sell: (Prezzo Entry - Prezzo Attuale) * Qty
            pnl = (price_entry - current_price) * qty

        value_eur = qty * current_price

        # Timestamp creazione
        created_at_dt = entry_candle.get('timestamp_dt')
        if created_at_dt is None:
             raw = entry_candle['timestamp']
             created_at_dt = datetime.strptime(raw, "%Y-%m-%d %H:%M:%S") if isinstance(raw, str) else raw

        created_at_str = created_at_dt.strftime("%Y-%m-%d %H:%M:%S")
        record_date_str = created_at_dt.strftime("%Y-%m-%d")


        # Costruzione Oggetto (Dizionario)
        fake_order = {
            "wallet_id": None,          # Escluso come richiesto
            "pair": pair_limits.get('pair'),
            "kr_pair": pair_limits.get('kr_pair'),
            "base": pair_limits.get('base'),
            "quote": pair_limits.get('quote'),
            "qty": qty,
            "price_entry": price_entry,
            "price_avg": price_entry,   # Assumiamo singola entrata
            "take_profit": take_profit,
            "stop_loss": stop_loss,
            "price": current_price,     # Prezzo attuale di mercato
            "value_eur": value_eur,
            "pnl": pnl,
            "type": type,         # Come da screen
            "subtype": subtype,
            "created_at": created_at_dt, # Il vectorizer gestisce datetime o stringhe solitamente
            "record_date": record_date_str,
            "status": "OPEN",
            "price_out": None,          # Escluso come richiesto
            "decision_id": None,         # Escluso come richiesto
            "lev": lev
        }

        return fake_order

    def generate_oracle_label(self, future_candles, current_price, wallet_balance, min_order_cost=10.0, pair_limits=None, fake_order=None):
            """
            Calcola i target ideali.
            Gestisce sia l'apertura (se no ordini) che la chiusura/management (se ordine attivo).
            """
            if not future_candles or len(future_candles) < 5:
                return None

            # ==============================================================================
            # CASO A: GESTIONE ORDINE ESISTENTE (Chiudere o Holdare)
            # ==============================================================================
            if fake_order:
                # Dati ordine esistente
                entry_price = float(fake_order['price_entry'])
                tp_price = float(fake_order['take_profit']) if fake_order['take_profit'] else None
                sl_price = float(fake_order['stop_loss']) if fake_order['stop_loss'] else None
                order_lev = float(fake_order['lev'])
                subtype = fake_order['subtype'] # 'buy' o 'sell'

                # Target defaults: HOLD
                target_side = 2
                target_qty = 0.0
                target_ordertype = 0 # Limit default

                should_close = False

                # --- LOGICA LONG (BUY) ---
                if subtype == 'buy':
                    # 1. Controllo Immediato (Siamo già fuori range?)
                    # Se prezzo attuale > TP (Riscuoti) o < SL (Stop Loss) -> Chiudi subito
                    if (tp_price and current_price >= tp_price) or (sl_price and current_price <= sl_price):
                        should_close = True

                    # 2. Controllo Futuro (Hit TP/SL nelle prossime ore)
                    if not should_close:
                        for c in future_candles:
                            if (tp_price and c['high'] >= tp_price) or (sl_price and c['low'] <= sl_price):
                                should_close = True
                                break

                    # 3. Random Profit Taking (20%)
                    # Se siamo in profitto ma non a TP, a volte chiudiamo comunque
                    if not should_close:
                        is_profitable = current_price > entry_price
                        if is_profitable and random.random() < 0.20:
                            should_close = True

                    # Azione
                    if should_close:
                        target_side = 1 # Chiudi Long -> SELL
                        target_qty = 1.0 # Chiudi tutto
                        target_ordertype = 1 # Market per uscire sicuro

                # --- LOGICA SHORT (SELL) ---
                elif subtype == 'sell':
                    # 1. Controllo Immediato
                    # Short: Profitto se prezzo scende (curr < TP), Loss se sale (curr > SL)
                    # Nota: In short il TP è più basso dell'entry, SL è più alto.
                    if (tp_price and current_price <= tp_price) or (sl_price and current_price >= sl_price):
                        should_close = True

                    # 2. Controllo Futuro
                    if not should_close:
                        for c in future_candles:
                            if (tp_price and c['low'] <= tp_price) or (sl_price and c['high'] >= sl_price):
                                should_close = True
                                break

                    # 3. Random Profit Taking (20%)
                    if not should_close:
                        is_profitable = current_price < entry_price
                        if is_profitable and random.random() < 0.20:
                            should_close = True

                    # Azione
                    if should_close:
                        target_side = 0 # Chiudi Short -> BUY
                        target_qty = 1.0
                        target_ordertype = 1 # Market

                # Restituzione tensori per Ordine Esistente
                return {
                    "side": torch.tensor([target_side], dtype=torch.long),
                    "qty": torch.tensor([target_qty], dtype=torch.float32).view(-1, 1),
                    # Offset, TP, SL non servono in chiusura, mettiamo 0
                    "px_offset": torch.tensor([0.0], dtype=torch.float32).view(-1, 1),
                    "tp_mult": torch.tensor([0.0], dtype=torch.float32).view(-1, 1),
                    "sl_mult": torch.tensor([0.0], dtype=torch.float32).view(-1, 1),
                    "ordertype": torch.tensor([target_ordertype], dtype=torch.long),
                    # La leva DEVE essere quella dell'ordine per coerenza
                    "leverage": torch.tensor([order_lev], dtype=torch.float32).view(-1, 1),
                    # Se holdiamo, halt_prob alto, se chiudiamo basso
                    "halt_prob": torch.tensor([1.0 if target_side==2 else 0.0], dtype=torch.float32).view(-1, 1)
                }


            # ==============================================================================
            # CASO B: NESSUN ORDINE (Logica Originale "Fresh Entry")
            # ==============================================================================

            # --- 1. CHECK POVERTA ---
            if wallet_balance < min_order_cost:
                return {
                    "side": torch.tensor([2], dtype=torch.long),
                    "qty": torch.tensor([0.0], dtype=torch.float32).view(-1, 1),
                    "px_offset": torch.tensor([0.0], dtype=torch.float32).view(-1, 1),
                    "tp_mult": torch.tensor([0.0], dtype=torch.float32).view(-1, 1),
                    "sl_mult": torch.tensor([0.0], dtype=torch.float32).view(-1, 1),
                    "ordertype": torch.tensor([0], dtype=torch.long),
                    "leverage": torch.tensor([1.0], dtype=torch.float32).view(-1, 1),
                    "halt_prob": torch.tensor([1.0], dtype=torch.float32).view(-1, 1)
                }

            # --- 2. ANALISI MERCATO ---
            MIN_PROFIT_PCT = 0.045
            MAX_STOP_LOSS_TOLERANCE = 0.020
            RISK_PER_TRADE = 0.02

            highs = [c['high'] for c in future_candles]
            lows = [c['low'] for c in future_candles]
            closes = [c['close'] for c in future_candles]

            # Default: HOLD
            target_side = 2
            target_qty = 0.0
            target_tp_mult = 0.0
            target_sl_mult = 0.0
            target_ordertype = 0
            target_leverage = 1.0
            target_halt = 0.95

            # Target per offset di prezzo (solo per LIMIT)
            target_px_offset = 0.0

            def _clamp(v, lo, hi):
                return max(lo, min(hi, v))

            def _estimate_atr_pct(hs, ls, cs):
                """Stima veloce di ATR% usando TR medio su una finestra breve."""
                if not hs or not ls or not cs:
                    return 0.002  # fallback 0.2%
                tr = []
                prev = cs[0]
                for h, l, c in zip(hs, ls, cs):
                    tr_i = max(h - l, abs(h - prev), abs(l - prev))
                    tr.append(tr_i)
                    prev = c
                atr = float(sum(tr)) / float(len(tr)) if tr else 0.0
                base = float(cs[0]) if float(cs[0]) > 0 else float(hs[0])
                if base <= 0:
                    return 0.002
                return _clamp(atr / base, 0.001, 0.03)  # 0.1% .. 3%

            def _compute_limit_entry_and_px(current_px, side, highs_all, lows_all, closes_all):
                """
                Calcola un entry price "da professionisti":
                - BUY: limite su supporto (pivot low / quantile dei min) sotto al prezzo corrente
                - SELL: limite su resistenza (pivot high / quantile dei max) sopra al prezzo corrente

                Restituisce (entry_price, px_offset_target).

                px_offset_target è normalizzato in [-1,1] assumendo che a runtime
                venga scalato circa su ±5% (tanh). Se il tuo executor usa un'altra
                scala, cambia PX_MAX_PCT.
                """
                PX_MAX_PCT = 0.05

                w = min(8, len(highs_all))
                hs = list(map(float, highs_all[:w]))
                ls = list(map(float, lows_all[:w]))
                cs = list(map(float, closes_all[:w]))

                atr_pct = _estimate_atr_pct(hs, ls, cs)
                buffer = 0.15 * atr_pct  # piccolo buffer per evitare di mettere il limit "a metà" del rumore

                if side == 0:  # BUY
                    # support candidates: min window + quantile 25% (più robusto a wick singoli)
                    sup_min = min(ls)
                    sup_q = float(np.quantile(ls, 0.25)) if len(ls) >= 4 else sup_min
                    support = max(min(sup_min, sup_q), 1e-9)
                    entry = support * (1.0 + buffer)
                    # deve stare sotto current (limit buy), altrimenti fallback a piccolo pullback
                    if entry >= current_px:
                        entry = current_px * (1.0 - max(0.002, 0.5 * atr_pct))
                else:  # SELL
                    res_max = max(hs)
                    res_q = float(np.quantile(hs, 0.75)) if len(hs) >= 4 else res_max
                    resistance = max(res_max, res_q)
                    entry = resistance * (1.0 - buffer)
                    # deve stare sopra current (limit sell), altrimenti fallback a piccolo pullback
                    if entry <= current_px:
                        entry = current_px * (1.0 + max(0.002, 0.5 * atr_pct))

                raw_offset = (entry - current_px) / current_px if current_px > 0 else 0.0
                raw_offset = _clamp(raw_offset, -PX_MAX_PCT, PX_MAX_PCT)
                px_norm = _clamp(raw_offset / PX_MAX_PCT, -1.0, 1.0)
                return float(entry), float(px_norm)

            # --- LOGICA BUY ---
            sl_threshold_price = current_price * (1 - MAX_STOP_LOSS_TOLERANCE)
            tp_threshold_price = current_price * (1 + MIN_PROFIT_PCT)

            idx_tp_hit = 9999
            idx_sl_hit = 9999

            try: idx_tp_hit = next(i for i, x in enumerate(highs) if x > tp_threshold_price)
            except: pass
            try: idx_sl_hit = next(i for i, x in enumerate(lows) if x < sl_threshold_price)
            except: pass

            if idx_tp_hit < idx_sl_hit:
                target_side = 0
                target_qty = 0.95

                # Se la prima candela futura è già "molto sopra" -> entra market, altrimenti prova limit su supporto
                if (closes[0] - current_price)/current_price > 0.003:
                    target_ordertype = 1  # MARKET
                    entry_price = float(current_price)
                    target_px_offset = 0.0
                else:
                    target_ordertype = 0  # LIMIT
                    entry_price, target_px_offset = _compute_limit_entry_and_px(
                        float(current_price), 0, highs, lows, closes
                    )

                # TP/SL e leva DEVONO dipendere dal prezzo di entry
                best_exit_price = max(highs[:idx_tp_hit+5])
                pct_gain = (best_exit_price - entry_price) / entry_price if entry_price > 0 else 0.0
                target_tp_mult = pct_gain / 0.10

                lowest_before_tp = min(lows[:idx_tp_hit+1])
                safe_sl_price = lowest_before_tp * 0.998
                pct_loss = (entry_price - safe_sl_price) / entry_price if entry_price > 0 else 0.0
                pct_loss = max(0.002, pct_loss)
                target_sl_mult = pct_loss / 0.05

                raw_lev = RISK_PER_TRADE / pct_loss
                max_pair_lev = float(pair_limits.get('leverage_buy_max', 1.0)) if pair_limits else 1.0
                safe_lev = min(raw_lev, max_pair_lev, 5.0)
                target_leverage = safe_lev
                target_halt = 0.05

            # --- LOGICA SELL ---
            elif (current_price - min(lows))/current_price > MIN_PROFIT_PCT:
                target_side = 1
                target_qty = 0.95

                # Se la prima candela futura è già "molto sotto" -> entra market, altrimenti prova limit su resistenza
                if (current_price - closes[0]) / current_price > 0.003:
                    target_ordertype = 1  # MARKET
                    entry_price = float(current_price)
                    target_px_offset = 0.0
                else:
                    target_ordertype = 0  # LIMIT
                    entry_price, target_px_offset = _compute_limit_entry_and_px(
                        float(current_price), 1, highs, lows, closes
                    )

                best_low = min(lows)
                pct_gain = (entry_price - best_low) / entry_price if entry_price > 0 else 0.0
                target_tp_mult = pct_gain / 0.10

                highest_before_low = max(highs)
                pct_loss = (highest_before_low - entry_price) / entry_price if entry_price > 0 else 0.0
                pct_loss = max(0.002, pct_loss)
                target_sl_mult = pct_loss / 0.05

                raw_lev = RISK_PER_TRADE / pct_loss
                max_pair_lev = float(pair_limits.get('leverage_sell_max', 1.0)) if pair_limits else 1.0
                safe_lev = min(raw_lev, max_pair_lev, 5.0)
                target_leverage = safe_lev
                target_halt = 0.05

            target_tp_mult = max(0.1, min(target_tp_mult, 5.0))
            target_sl_mult = max(0.1, min(target_sl_mult, 5.0))

            return {
                "side": torch.tensor([target_side], dtype=torch.long),
                "qty": torch.tensor([target_qty], dtype=torch.float32).view(-1, 1),
                "px_offset": torch.tensor([target_px_offset], dtype=torch.float32).view(-1, 1),
                "tp_mult": torch.tensor([target_tp_mult], dtype=torch.float32).view(-1, 1),
                "sl_mult": torch.tensor([target_sl_mult], dtype=torch.float32).view(-1, 1),
                "ordertype": torch.tensor([target_ordertype], dtype=torch.long),
                "leverage": torch.tensor([target_leverage], dtype=torch.float32).view(-1, 1),
                "halt_prob": torch.tensor([target_halt], dtype=torch.float32).view(-1, 1)
            }

    def train_step(self, context, pair_limits, future_candles, current_step_idx):
        self.model.train()
        # NOTA: Non azzeriamo i gradienti qui! Lo facciamo solo dopo l'accumulo.

        # 0. Generazione Ordini Finti (Augmentation)
        # Se non c'è un ordine, proviamo a generarne uno finto per insegnare al modello a gestire posizioni aperte
        fake_order = self.generate_fake_order(context, pair_limits)
        if fake_order:
            context['order'] = fake_order

        # 1. Simulazione Wallet (Uguale)
        if random.random() < 0.2:
            simulated_wallet = random.uniform(0.1, 9.0)
        else:
            simulated_wallet = random.uniform(20.0, 5000.0)

        inputs, ref_price = self.vectorizer.vectorize(
            candles_db_data=context['candles'],
            open_order=context['order'], # Passiamo l'ordine (reale o finto)
            forecast_db_data=context['forecast'],
            pair_limits=pair_limits,
            wallet_balance=simulated_wallet
        )

        device = next(self.model.parameters()).device
        # Assegniamo i pesi solo alla loss del Side
        if self.loss_ce_side.weight.device != device:
             self.loss_ce_side.weight = self.weights_side.to(device)

        for k, v in inputs.items():
            inputs[k] = v.to(device)

        # 2. Labeling (come prima, targets una volta sola)
        current_close = context['candles']['1h'][-1]['close']
        targets = self.generate_oracle_label(
            future_candles,
            current_close,
            simulated_wallet,
            10.0,
            pair_limits,
            fake_order
        )

        if targets is None:
            return None

        t_side = targets['side'].to(device)
        t_qty  = targets['qty'].to(device)
        t_px   = targets['px_offset'].to(device)
        t_tp   = targets['tp_mult'].to(device)
        t_sl   = targets['sl_mult'].to(device)
        t_type = targets['ordertype'].to(device)
        t_lev  = targets['leverage'].to(device)
        t_halt = targets['halt_prob'].to(device)

        # 3. Loss Calcolo MULTISTEP
        is_active = (t_side != 2).float().view(-1, 1)

        h = None
        sum_w = 0.0

        # Accumulatori inizializzati come tensori sul device
        acc_loss_side = torch.tensor(0.0, device=device)
        acc_loss_qty  = torch.tensor(0.0, device=device)
        acc_loss_tp   = torch.tensor(0.0, device=device)
        acc_loss_sl   = torch.tensor(0.0, device=device)
        acc_loss_px   = torch.tensor(0.0, device=device)
        acc_loss_lev  = torch.tensor(0.0, device=device)
        acc_loss_type = torch.tensor(0.0, device=device)
        acc_loss_halt = torch.tensor(0.0, device=device)

        last_preds = None

        for s in range(self.thinking_steps):
            # Peso: diamo piu importanza agli step finali
            w = float(s + 1) / float(self.thinking_steps)
            sum_w += w

            # Forward con memoria ricorrente del cervello centrale
            y, h = self.model(inputs, h)
            preds = self.model.get_heads_dict(y)
            last_preds = preds  # per logging a fine funzione

            # Loss per questo step (stessa logica di prima)
            loss_side_step = self.loss_ce_side(preds['side'], t_side).squeeze()

            # queste hanno mask is_active -> diventano [B,1], le riduciamo con mean()
            loss_qty_step = (self.loss_mse(preds['qty'], t_qty) * is_active).mean().squeeze()
            loss_tp_step  = (self.loss_mse(preds['tp_mult'], t_tp) * is_active).mean().squeeze()
            loss_sl_step  = (self.loss_mse(preds['sl_mult'], t_sl) * is_active).mean().squeeze()
            loss_px_step  = (self.loss_mse(preds['price_offset'], t_px) * is_active).mean().squeeze()

            loss_lev_step  = self.loss_mse(preds['leverage'], t_lev).squeeze()          # scalare
            loss_type_step = self.loss_ce_type(preds['ordertype'], t_type).squeeze()    # scalare
            # loss_halt_step = self.loss_bce(preds['halt_prob'], t_halt).squeeze()        # scalare

            # ---- HALT focal BCE (stabile su class imbalance) ----
            p = preds['halt_prob'].clamp(1e-4, 1.0 - 1e-4)   # evita log(0)
            # ---- HALT target schedule per-step ----
            # Allinea il training al "thinking loop": primi step => target halt più basso (spinge a pensare),
            # ultimi step => converge al target finale (HOLD alto / TRADE basso).
            t_final = t_halt.float()  # ~0.95 (HOLD) o ~0.05 (trade)

            progress = float(s + 1) / float(self.thinking_steps)  # 0..1
            HALT_START_HOLD = 0.20
            HALT_START_TRADE = 0.02

            t_start = torch.where(
                t_final >= 0.5,
                torch.full_like(t_final, HALT_START_HOLD),
                torch.full_like(t_final, HALT_START_TRADE)
            )

            t = (t_start + (t_final - t_start) * progress).clamp(0.0, 1.0)
            # BCE per-sample
            bce = -(t * torch.log(p) + (1.0 - t) * torch.log(1.0 - p))

            # pt = prob della classe corretta
            pt = torch.where(t >= 0.5, p, 1.0 - p)

            # alpha bilanciato (HOLD vs non-HOLD)
            alpha = torch.where(t >= 0.5,
                                torch.full_like(t, self.halt_alpha_pos),
                                torch.full_like(t, self.halt_alpha_neg))

            loss_halt_step = (alpha * (1.0 - pt) ** self.halt_gamma * bce).mean()

            # scala (così halt non domina side/qty ecc.)
            loss_halt_step = self.halt_loss_weight * loss_halt_step

            # Accumulo pesato
            acc_loss_side = acc_loss_side + (w * loss_side_step)
            acc_loss_qty  = acc_loss_qty  + (w * loss_qty_step)
            acc_loss_tp   = acc_loss_tp   + (w * loss_tp_step)
            acc_loss_sl   = acc_loss_sl   + (w * loss_sl_step)
            acc_loss_px   = acc_loss_px   + (w * loss_px_step)
            acc_loss_lev  = acc_loss_lev  + (w * loss_lev_step)
            acc_loss_type = acc_loss_type + (w * loss_type_step)
            acc_loss_halt = acc_loss_halt + (w * loss_halt_step)

        # Media pesata tra gli step
        loss_side = acc_loss_side / sum_w
        loss_qty  = acc_loss_qty  / sum_w
        loss_tp   = acc_loss_tp   / sum_w
        loss_sl   = acc_loss_sl   / sum_w
        loss_px   = acc_loss_px   / sum_w
        loss_lev  = acc_loss_lev  / sum_w
        loss_type = acc_loss_type / sum_w
        loss_halt = acc_loss_halt / sum_w

        # 4. Loss totale (stessi pesi di prima)
        total_loss = (
            3.0 * loss_side +
            1.0 * loss_qty +
            0.5 * loss_tp +
            0.5 * loss_sl +
            0.3 * loss_lev +
            0.3 * loss_type +  # peso leggermente maggiore per ordertype
            0.1 * loss_px +
            0.4 * loss_halt
        )

        # --- GESTIONE GRADIENT ACCUMULATION ---

        # Normalizziamo la loss (perche sommeremo i gradienti 4 volte)
        loss_normalized = total_loss / self.accumulation_steps
        loss_normalized.backward()  # Accumula il gradiente

        # Se abbiamo raggiunto il numero di step o siamo alla fine, aggiorniamo
        if (current_step_idx + 1) % self.accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()  # Resetta ORA, dopo l'update

            # Aggiorna scheduler (opzionale farlo qui o a fine epoca)
            self.scheduler.step(total_loss)

        return {
            "loss": total_loss.item(),  # Ritorniamo la loss vera (non normalizzata) per i log
            "target_side": t_side.item(),
            "pred_side": torch.argmax(last_preds['side']).item() if last_preds is not None else -1
        }

    def save_checkpoint(self, path="model_checkpoint.pth"):
        torch.save(self.model.state_dict(), path)
        # print(f"--- Saved {path} ---")
