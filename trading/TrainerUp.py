import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import zlib
import os
import numpy as np
from datetime import timedelta, datetime
from trading.RLReward import PnlRewardManager

class TradingTrainer:
    def __init__(self, model, db_manager, vectorizer, learning_rate=2e-5):
        self.model = model
        self.db = db_manager
        self.vectorizer = vectorizer

        # --- HALT tuning ---
        self.halt_gamma = 2.0          # focal gamma
        self.halt_alpha_pos = 0.6      # peso quando target_halt ~ 1 (HOLD)
        self.halt_alpha_neg = 0.4      # peso quando target_halt ~ 0 (BUY/SELL)
        self.halt_loss_weight = 0.6    # quanto pesa halt nella loss totale

        # --- RL Reward Manager ---
        self.reward_manager = PnlRewardManager()
        self.rl_weight = 0.5 # Peso della loss RL rispetto alla supervisionata


        # Carica pesi (Best effort)
        try:
        #     self.model.load_state_dict(torch.load("trm_model_v2.pth"), strict=False)
            self.model.load_state_dict(torch.load("model/trainerUpPnlHalt.pth"), strict=False)
            print("--- Pesi 'Best Model' caricati ---")
        except:
            print("--- Nessun peso precedente, start fresh ---")

        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)

        # --- SCHEDULER ---
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=50
        )
        self.accumulation_steps = 8

        # --- MULTISTEP THINKING ---
        # Numero di step di "pensiero" durante il training (unroll della GRUCell del cervello)
        self.thinking_steps = 5  # Ridotto da 7 per stabilità gradienti

        # --- LOSS FUNCTIONS SEPARATE ---

        # 1. Loss per il SIDE (3 classi: Buy, Sell, Hold)
        # 2. Bilanciamento Classi (HOLD è molto frequente, alziamo pesi BUY/SELL)
        # Portiamo da [1.8, 1.8, 1.0] a [3.0, 3.0, 1.0] per forzare uscita da HOLD
        self.weights_side = torch.tensor([2.2, 2.2, 1.0])
        self.criterion_side = nn.CrossEntropyLoss(weight=self.weights_side)

        # 2. Loss per ORDER TYPE (2 classi: Limit, Market)
        # Qui NON usiamo pesi (o standard), perche Limit e Market sono bilanciati
        self.loss_ce_type = nn.CrossEntropyLoss()

        self.loss_mse = nn.MSELoss()
        self.loss_bce = nn.BCELoss()

    def _compute_clarity_score(self, pnl_buy: float, pnl_sell: float, pnl_hold: float) -> float:
        """
        Calcola quanto è "chiara" la decisione migliore basandosi sul margine tra la miglior azione e la seconda.
        """
        pnl_values = [pnl_buy, pnl_sell, pnl_hold]
        sorted_pnl = sorted(pnl_values, reverse=True)
        best = sorted_pnl[0]
        second_best = sorted_pnl[1]

        margin = best - second_best
        normalized_margin = margin / 0.02 # 2% = "chiaro"

        import math
        clarity = 1.0 / (1.0 + math.exp(-3.0 * (normalized_margin - 1.0)))
        return max(0.1, min(0.95, clarity))

    def _compute_smart_close_probability(
        self, entry_price: float, current_price: float,
        tp_price: float, sl_price: float,
        future_candles: list, subtype: str, order_lev: float = 1.0
    ) -> float:
        """
        Calcola probabilità di chiusura smart basata su:
        1. PnL% effettivo (quanto profitto hai fatto in %)
        2. Progress verso il Take Profit (quanto sei vicino al target)
        3. Momentum/Reversal detection (il mercato sta girando?)
        4. Future Risk (sta per andare contro di te?)

        Returns: probabilità 0.0-1.0 di chiudere l'ordine.
        """
        import math

        # === CALCOLO PNL% ===
        if subtype == 'buy':
            pnl_pct = (current_price - entry_price) / entry_price
        else:  # sell
            pnl_pct = (entry_price - current_price) / entry_price

        # Considera la leva per il PnL effettivo sul margine
        effective_pnl_pct = pnl_pct * order_lev

        # Se in perdita, non chiudere anticipatamente (lascia che colpisca SL)
        if effective_pnl_pct <= 0:
            return 0.0

        # === SOGLIE (Aggiornate per coprire le fee) ===
        MIN_PNL_PCT = 0.025       # 2.5% minimo (Fee su 5x leva ~2.0%)
        GOOD_PNL_PCT = 0.05       # 5% = profitto decente
        HIGH_PNL_PCT = 0.10       # 10% = profitto alto
        VERY_HIGH_PNL_PCT = 0.20  # 20% = profitto eccellente

        # Se PnL troppo basso, non chiudere (non vale la pena rischiare per 10 centesimi)
        if effective_pnl_pct < MIN_PNL_PCT:
            return 0.0

        # === FATTORE 1: PNL% ===
        # Sigmoid centrata a 6% (GOOD + 1%), scala più morbida
        pnl_sigmoid_input = (effective_pnl_pct - 0.06) / 0.04
        pnl_factor = 0.05 + 0.65 * (1.0 / (1.0 + math.exp(-pnl_sigmoid_input)))

        # === FATTORE 2: PROGRESS TO TP (0.0 - 0.25) ===
        progress_factor = 0.0
        if tp_price and tp_price > 0:
            if subtype == 'buy':
                total_distance = tp_price - entry_price
                current_progress = current_price - entry_price
            else:  # sell
                total_distance = entry_price - tp_price
                current_progress = entry_price - current_price

            if total_distance > 0:
                progress_ratio = current_progress / total_distance
                progress_ratio = max(0.0, min(1.0, progress_ratio))

                if progress_ratio > 0.6:
                    progress_factor = 0.25 * ((progress_ratio - 0.6) / 0.4)

        # === FATTORE 3: MOMENTUM / REVERSAL DETECTION (-0.15 - +0.25) ===
        momentum_factor = 0.0
        if future_candles and len(future_candles) >= 3:
            look_ahead = min(5, len(future_candles))

            if subtype == 'buy':
                future_closes = [float(c['close']) for c in future_candles[:look_ahead]]
                future_lows = [float(c['low']) for c in future_candles[:look_ahead]]

                avg_future = sum(future_closes) / len(future_closes)
                direction = (avg_future - current_price) / current_price

                if direction < -0.005:
                    momentum_factor = min(0.25, abs(direction) * 10)
                elif direction > 0.01:
                    momentum_factor = -0.15

                if any(low < entry_price for low in future_lows):
                    momentum_factor += 0.20

            else:  # sell
                future_closes = [float(c['close']) for c in future_candles[:look_ahead]]
                future_highs = [float(c['high']) for c in future_candles[:look_ahead]]

                avg_future = sum(future_closes) / len(future_closes)
                direction = (avg_future - current_price) / current_price

                if direction > 0.005:
                    momentum_factor = min(0.25, direction * 10)
                elif direction < -0.01:
                    momentum_factor = -0.15

                if any(high > entry_price for high in future_highs):
                    momentum_factor += 0.20

        # === FATTORE 4: FUTURE RISK (0.0 - 0.30) ===
        future_risk_factor = 0.0
        if future_candles and len(future_candles) >= 5 and sl_price:
            if subtype == 'buy':
                future_lows = [float(c['low']) for c in future_candles[:8]]
                min_future_low = min(future_lows) if future_lows else current_price

                if sl_price > 0:
                    distance_to_sl = (current_price - sl_price) / current_price
                    future_distance_to_sl = (min_future_low - sl_price) / current_price if current_price > 0 else 0

                    if future_distance_to_sl < distance_to_sl * 0.5:
                        future_risk_factor = 0.20
                    if min_future_low <= sl_price:
                        future_risk_factor = 0.30
            else:  # sell
                future_highs = [float(c['high']) for c in future_candles[:8]]
                max_future_high = max(future_highs) if future_highs else current_price

                if sl_price > 0:
                    distance_to_sl = (sl_price - current_price) / current_price
                    future_distance_to_sl = (sl_price - max_future_high) / current_price if current_price > 0 else 0

                    if future_distance_to_sl < distance_to_sl * 0.5:
                        future_risk_factor = 0.20
                    if max_future_high >= sl_price:
                        future_risk_factor = 0.30

        # === COMBINA I FATTORI ===
        total_probability = pnl_factor + progress_factor + momentum_factor + future_risk_factor
        total_probability = max(0.0, min(0.95, total_probability))

        # === OVERRIDE PER PNL MOLTO ALTI ===
        if effective_pnl_pct >= VERY_HIGH_PNL_PCT:
            total_probability = max(total_probability, 0.85)
        elif effective_pnl_pct >= HIGH_PNL_PCT:
            total_probability = max(total_probability, 0.60)
        elif effective_pnl_pct >= GOOD_PNL_PCT:
            total_probability = max(total_probability, 0.35)

        return total_probability

    def _compute_pnl_reward(self, preds, current_price, future_candles, fee_rate=0.001):
        """
        Compute simulated PnL for a given prediction.
        Returns a dictionary with PnL and debug info.
        """
        if not future_candles or len(future_candles) < 5:
            return {'pnl': 0.0, 'hit_tp': 0, 'hit_sl': 0}

        side = int(torch.argmax(preds["side"]).item())
        if side == 2:  # HOLD
            return {'pnl': 0.0, 'hit_tp': 0, 'hit_sl': 0}

        # Extract prediction values
        qty_frac = float(preds["qty"].clamp(0, 1).item())
        lev = float(preds["leverage"].clamp(1, 5).item())
        tp_mult = float(preds["tp_mult"].clamp(0, 10).item())
        sl_mult = float(preds["sl_mult"].clamp(0, 10).item())

        notional = 100.0 * qty_frac * lev
        entry = float(current_price)

        if notional <= 0.0 or entry <= 0.0:
            return {'pnl': 0.0, 'hit_tp': 0, 'hit_sl': 0}

        # TP/SL percentages (simplified, consistent with Decoder)
        tp_pct = max(0.004, min(tp_mult * 0.10, 0.50))
        sl_pct = max(0.003, min(sl_mult * 0.05, 0.50))

        hit_tp = 0
        hit_sl = 0

        # Simulate exit price based on TP/SL hitting
        if side == 0:  # BUY
            tp = entry * (1.0 + tp_pct)
            sl = entry * (1.0 - sl_pct)
            exit_px = float(future_candles[-1]["close"])
            for c in future_candles:
                if float(c["low"]) <= sl:
                    exit_px = sl
                    hit_sl = 1
                    break
                if float(c["high"]) >= tp:
                    exit_px = tp
                    hit_tp = 1
                    break
            raw_ret = (exit_px - entry) / entry
        else:  # SELL
            tp = entry * (1.0 - tp_pct)
            sl = entry * (1.0 + sl_pct)
            exit_px = float(future_candles[-1]["close"])
            for c in future_candles:
                if float(c["high"]) >= sl:
                    exit_px = sl
                    hit_sl = 1
                    break
                if float(c["low"]) <= tp:
                    exit_px = tp
                    hit_tp = 1
                    break
            raw_ret = (entry - exit_px) / entry

        # HighTF fee rate might be slightly different or we keep 0.001
        net_ret = raw_ret - 2.0 * fee_rate
        return {
            'pnl': float(notional * net_ret),
            'hit_tp': hit_tp,
            'hit_sl': hit_sl,
            'qty_frac': qty_frac,
            'lev': lev,
            'tp_mult': tp_mult,
            'sl_mult': sl_mult
        }

    def _simulate_clarity_pnl(self, side: int, curr_price: float, futures: list,
                               tp_mult: float = 0.5, sl_mult: float = 0.5) -> float:
        """
        Simula il PnL per una data azione (BUY, SELL o HOLD) usando la stessa logica
        dell'oracle avanzato (_compute_pnl_reward).
        Implementazione basata su TP/SL e traiettoria completa (high/low).
        """
        if side == 2:  # HOLD
            return 0.0
        if not futures or len(futures) < 5:
            return 0.0

        entry = float(curr_price)
        # TP/SL costanti per la misura di "chiarezza" (default conservativi)
        tp_pct = max(0.004, min(tp_mult * 0.10, 0.50))
        sl_pct = max(0.003, min(sl_mult * 0.05, 0.50))

        if side == 0:  # BUY
            tp = entry * (1.0 + tp_pct)
            sl = entry * (1.0 - sl_pct)
            exit_px = float(futures[-1]['close'])
            for c in futures:
                if float(c['low']) <= sl:
                    exit_px = sl
                    break
                if float(c['high']) >= tp:
                    exit_px = tp
                    break
            raw_ret = (exit_px - entry) / entry
        else:  # SELL
            tp = entry * (1.0 - tp_pct)
            sl = entry * (1.0 + sl_pct)
            exit_px = float(futures[-1]['close'])
            for c in futures:
                if float(c['high']) >= sl:
                    exit_px = sl
                    break
                if float(c['low']) <= tp:
                    exit_px = tp
                    break
            raw_ret = (entry - exit_px) / entry

        # Sottraiamo fee_rate round-trip (approssimato 0.1% * 2)
        net_ret = raw_ret - 0.002
        return float(net_ret)

    def _compute_rl_loss_multi_head(self, preds, step_reward, pred_side_idx, device):
        """
        Compute RL losses for ALL heads, not just side.
        """
        log_probs_side = F.log_softmax(preds['side'], dim=-1)
        picked_lp_side = log_probs_side.gather(1, pred_side_idx.view(-1, 1)).view(-1)
        rl_side = -(step_reward * picked_lp_side).mean()

        rl_qty = -(step_reward * preds['qty']).mean() * 0.1
        rl_tp  = -(step_reward * preds['tp_mult']).mean() * 0.1
        rl_sl  = -(step_reward * preds['sl_mult']).mean() * 0.1
        rl_lev = -(step_reward * preds['leverage']).mean() * 0.1

        return rl_side + rl_qty + rl_tp + rl_sl + rl_lev

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

                    # 3. SMART Profit Taking (basato su PnL%, momentum, future risk)
                    # Usa algoritmo intelligente invece della probabilità fissa 20%
                    if not should_close:
                        close_prob = self._compute_smart_close_probability(
                            entry_price=entry_price,
                            current_price=current_price,
                            tp_price=tp_price,
                            sl_price=sl_price,
                            future_candles=future_candles,
                            subtype='buy',
                            order_lev=order_lev
                        )
                        if close_prob > 0 and random.random() < close_prob:
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

                    # 3. SMART Profit Taking (basato su PnL%, momentum, future risk)
                    # Usa algoritmo intelligente invece della probabilità fissa 20%
                    if not should_close:
                        close_prob = self._compute_smart_close_probability(
                            entry_price=entry_price,
                            current_price=current_price,
                            tp_price=tp_price,
                            sl_price=sl_price,
                            future_candles=future_candles,
                            subtype='sell',
                            order_lev=order_lev
                        )
                        if close_prob > 0 and random.random() < close_prob:
                            should_close = True

                    # Azione
                    if should_close:
                        target_side = 0 # Chiudi Short -> BUY
                        target_qty = 1.0
                        target_ordertype = 1 # Market

                # === NUOVO: Calcola Clarity Score per halt_prob ===
                pnl_buy = self._simulate_clarity_pnl(0, current_price, future_candles, 0.5, 0.5)
                pnl_sell = self._simulate_clarity_pnl(1, current_price, future_candles, 0.5, 0.5)
                pnl_hold = 0.0
                target_halt_val = self._compute_clarity_score(pnl_buy, pnl_sell, pnl_hold)

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
                    "halt_prob": torch.tensor([target_halt_val], dtype=torch.float32).view(-1, 1)
                }


            # ==============================================================================
            # CASO B: NESSUN ORDINE (Logica Semplificata per predizione direzione)
            # ==============================================================================

            # --- 1. CHECK POVERTA ---
            if wallet_balance < min_order_cost:
                return {
                    "side": torch.tensor([2], dtype=torch.long),
                    "qty": torch.tensor([0.0], dtype=torch.float32).view(-1, 1),
                    "px_offset": torch.tensor([0.0], dtype=torch.float32).view(-1, 1),
                    "tp_mult": torch.tensor([0.0], dtype=torch.float32).view(-1, 1),
                    "sl_mult": torch.tensor([0.0], dtype=torch.float32).view(-1, 1),
                    "ordertype": torch.tensor([1], dtype=torch.long),  # MARKET
                    "leverage": torch.tensor([1.0], dtype=torch.float32).view(-1, 1),
                    "halt_prob": torch.tensor([1.0], dtype=torch.float32).view(-1, 1)
                }

            # --- 2. ANALISI MERCATO (LOGICA SEMPLIFICATA PER DIREZIONE) ---
            LOOKAHEAD_CANDLES = 24  # Guarda N candele avanti (per 1h = 24 ore)

            # Assicurati di avere abbastanza candele
            if len(future_candles) < LOOKAHEAD_CANDLES:
                return None

            highs = [float(c['high']) for c in future_candles]
            lows = [float(c['low']) for c in future_candles]
            closes = [float(c['close']) for c in future_candles]

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

            # Prendi il close della candela N passi avanti
            future_close = float(closes[LOOKAHEAD_CANDLES - 1])

            # Default: HOLD (questo modello predice SOLO la direzione)
            target_side = 2
            target_qty = 0.0
            target_tp_mult = 0.0
            target_sl_mult = 0.0
            target_ordertype = 1  # Sempre MARKET (questo modello non piazza ordini)
            target_leverage = 1.0  # Sempre 1 (questo modello predice solo direzione)
            target_px_offset = 0.0  # Sempre 0 (non usato)

            # Calcola soglia dinamica basata su ATR% della currency
            atr_pct = _estimate_atr_pct(highs[:8], lows[:8], closes[:8])
            # 4. Profitability Filter (Oracle)
            # Abbassiamo leggermente da 1.1x ATR a 0.8x ATR per catturare più opportunità
            min_profit_threshold = max(0.010, 1 * atr_pct)

            # Calcola la variazione percentuale
            pct_change = (future_close - current_price) / current_price

            if pct_change >= min_profit_threshold:
                # BUY: il prezzo salirà
                target_side = 0
                target_qty = 0.95

                # TP/SL semplificati (per coerenza con le altre heads)
                best_high = max(highs[:LOOKAHEAD_CANDLES])
                worst_low = min(lows[:LOOKAHEAD_CANDLES])

                pct_gain = (best_high - current_price) / current_price if current_price > 0 else 0.0
                target_tp_mult = _clamp(pct_gain / 0.10, 0.1, 5.0)

                pct_loss = (current_price - worst_low) / current_price if current_price > 0 else 0.0
                pct_loss = max(0.002, pct_loss)
                target_sl_mult = _clamp(pct_loss / 0.05, 0.1, 5.0)

            elif pct_change <= -min_profit_threshold:
                # SELL: il prezzo scenderà
                target_side = 1
                target_qty = 0.95

                # TP/SL semplificati
                best_low = min(lows[:LOOKAHEAD_CANDLES])
                worst_high = max(highs[:LOOKAHEAD_CANDLES])

                pct_gain = (current_price - best_low) / current_price if current_price > 0 else 0.0
                target_tp_mult = _clamp(pct_gain / 0.10, 0.1, 5.0)

                pct_loss = (worst_high - current_price) / current_price if current_price > 0 else 0.0
                pct_loss = max(0.002, pct_loss)
                target_sl_mult = _clamp(pct_loss / 0.05, 0.1, 5.0)

            # else: HOLD (già impostato di default)

            target_tp_mult = max(0.1, min(target_tp_mult, 5.0))
            target_sl_mult = max(0.1, min(target_sl_mult, 5.0))

            pnl_buy = self._simulate_clarity_pnl(0, current_price, future_candles, 0.5, 0.5)
            pnl_sell = self._simulate_clarity_pnl(1, current_price, future_candles, 0.5, 0.5)
            target_halt = self._compute_clarity_score(pnl_buy, pnl_sell, 0.0)

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

        # --- BACKUP STATO RNG ---
        self._rng_state_backup = random.getstate()

        # --- DETERMINISTIC SEEDING ---
        try:
            pivot_candle = context['candles']['1h'][-1]
            ts_str = str(pivot_candle.get('timestamp_dt') or pivot_candle.get('timestamp'))
            pair_name = str(pair_limits.get('pair', 'unk'))

            # Seed STABILE cross-run
            unique_str = f"{pair_name}_{ts_str}"
            seed_val = zlib.adler32(unique_str.encode("utf-8")) & 0xffffffff
            random.seed(seed_val)
        except Exception:
            pass

        # NOTA: Non azzeriamo i gradienti qui! Lo facciamo solo dopo l'accumulo.

        try:
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
            if self.criterion_side.weight.device != device:
                 self.criterion_side.weight = self.weights_side.to(device)

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
            acc_loss_rl   = torch.tensor(0.0, device=device)

            last_preds = None

            for s in range(self.thinking_steps):
                # Peso ESPONENZIALE: favorendo gli step finali si forza il modello a "pensare"
                w = (float(s + 1) / float(self.thinking_steps)) ** 2
                sum_w += w

                # Forward con memoria ricorrente del cervello centrale
                y, h = self.model(inputs, h)
                preds = self.model.get_heads_dict(y)
                last_preds = preds  # per logging a fine funzione

                # Loss per questo step (stessa logica di prima)
                loss_side_step = self.criterion_side(preds['side'], t_side).squeeze()

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
                t_final = t_halt.float()  # clarity score [0.1, 0.95]
                progress = float(s + 1) / float(self.thinking_steps)  # 0..1

                # Convergenza basata sulla clarity
                speed = 0.5 + 1.5 * t_final.clamp(0.1, 0.95)
                effective_progress = (progress * speed).clamp(0.0, 1.0)

                HALT_START = 0.10
                t = (torch.full_like(t_final, HALT_START) + (t_final - HALT_START) * effective_progress).clamp(0.0, 1.0)
                # BCE per-sample
                bce = -(t * torch.log(p) + (1.0 - t) * torch.log(1.0 - p))

                # ---- Nuova Focal Loss continua per soft-targets ----
                pt_high = p
                pt_low = 1.0 - p
                pt = t * pt_high + (1.0 - t) * pt_low

                alpha = t * self.halt_alpha_pos + (1.0 - t) * self.halt_alpha_neg

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

                # ---- RL REWARD: TRUE PnL-BASED ----
                with torch.no_grad():
                    pred_side_idx = torch.argmax(preds['side'], dim=-1)

                    # Simulate PnL for this prediction
                    reward_info = self._compute_pnl_reward(
                        preds, current_close, future_candles
                    )
                    pnl_reward = reward_info['pnl']

                    # Scale PnL to reward: divide by 1.0 (HighTF has smaller expected returns per trade)
                    # or stay consistent with Trainer.py (divide by 10)
                    step_reward = torch.tensor(
                        max(-2.0, min(2.0, pnl_reward / 10.0)),
                        device=device
                    )

                # Multi-Head RL Loss
                loss_rl_step = self._compute_rl_loss_multi_head(preds, step_reward, pred_side_idx, device)
                acc_loss_rl = acc_loss_rl + (w * loss_rl_step)

            # Media pesata tra gli step
            loss_side = acc_loss_side / sum_w
            loss_qty  = acc_loss_qty  / sum_w
            loss_tp   = acc_loss_tp   / sum_w
            loss_sl   = acc_loss_sl   / sum_w
            loss_px   = acc_loss_px   / sum_w
            loss_lev  = acc_loss_lev  / sum_w
            loss_type = acc_loss_type / sum_w
            loss_halt = acc_loss_halt / sum_w
            loss_rl   = acc_loss_rl   / sum_w

            # 4. Loss totale (stessi pesi di prima)
            total_loss = (
                3.0 * loss_side +
                1.0 * loss_qty +
                0.5 * loss_tp +
                0.5 * loss_sl +
                0.3 * loss_lev +
                0.3 * loss_type +  # peso leggermente maggiore per ordertype
                0.1 * loss_px +
                0.4 * loss_halt +
                self.rl_weight * loss_rl
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
                # NOTA: Lo scheduler viene chiamato a fine epoca in RunTrainingGpuUp.py

            return {
                "loss": total_loss.item(),
                "loss_rl": loss_rl.item(),
                "loss_side": loss_side.item(),
                "loss_qty": loss_qty.item(),
                "loss_tp": loss_tp.item(),
                "loss_sl": loss_sl.item(),
                "loss_px": loss_px.item(),
                "loss_lev": loss_lev.item(),
                "loss_type": loss_type.item(),
                "loss_halt": loss_halt.item(),
                "target_side": t_side.item(),
                "pred_side": torch.argmax(last_preds['side']).item() if last_preds is not None else -1,
                # RL Debug Info
                "rl_pnl": reward_info['pnl'],
                "rl_hit_tp": reward_info['hit_tp'],
                "rl_hit_sl": reward_info['hit_sl']
            }

        finally:
            # --- RIPRISTINO STATO RNG ---
            if hasattr(self, '_rng_state_backup'):
                random.setstate(self._rng_state_backup)


    def save_checkpoint(self, path="model/trainerUpN.pth"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        # print(f"--- Saved {path} ---")
