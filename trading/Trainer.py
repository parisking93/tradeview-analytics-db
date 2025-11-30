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

        # Carica pesi se esistono
        # try:
        #     self.model.load_state_dict(torch.load("trm_model_v1.pth"))
        #     print("--- Pesi precedenti caricati ---")
        # except:
        #     print("--- Nessun peso precedente trovato, start fresh ---")

        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)

        # --- LOSS FUNCTIONS ---
        self.loss_ce = nn.CrossEntropyLoss()  # Per Side (0,1,2) e OrderType (0,1)
        self.loss_mse = nn.MSELoss()          # Per valori continui (Price, Qty, TP, SL, Lev)
        self.loss_bce = nn.BCELoss()          # Per probabilità pure (Halt)

    def generate_oracle_label(self, future_candles, current_price, wallet_balance, min_order_cost=10.0):
        """
        Calcola i target ideali per TUTTE le head del modello.
        """
        if not future_candles or len(future_candles) < 5:
            return None

        # --- 1. CHECK POVERTÀ (Resource Constraint) ---
        if wallet_balance < min_order_cost:
            # Se non ci sono soldi, il target è HOLD e tutto a zero
            return {
                "side": torch.tensor([2], dtype=torch.long),
                "qty": torch.tensor([0.0], dtype=torch.float32).view(-1, 1),
                "px_offset": torch.tensor([0.0], dtype=torch.float32).view(-1, 1),
                "tp_mult": torch.tensor([0.0], dtype=torch.float32).view(-1, 1),
                "sl_mult": torch.tensor([0.0], dtype=torch.float32).view(-1, 1),
                "ordertype": torch.tensor([0], dtype=torch.long), # Limit default
                "leverage": torch.tensor([0.0], dtype=torch.float32).view(-1, 1), # 0.0 -> Leva 1.0 (softplus)
                "halt": torch.tensor([1.0], dtype=torch.float32).view(-1, 1) # Sicuro di stare fermo
            }

        # --- 2. ANALISI MERCATO ---
        MIN_PROFIT_PCT = 0.025  # 2.5% target minimo
        MAX_STOP_LOSS_TOLERANCE = 0.020# Max SL tollerato 2%

        highs = [c['high'] for c in future_candles]
        lows = [c['low'] for c in future_candles]
        closes = [c['close'] for c in future_candles] # Utile per vedere chiusure immediate

        # Picchi assoluti nel futuro
        max_high = max(highs)
        min_low = min(lows)

        # Variazioni massime possibili
        max_up_pct = (max_high - current_price) / current_price
        max_down_pct = (current_price - min_low) / current_price

        # --- DEFAULT: HOLD ---
        target_side = 2
        target_qty = 0.0
        target_tp_mult = 0.0
        target_sl_mult = 0.0
        target_ordertype = 0 # 0=Limit, 1=Market
        target_leverage = 0.0 # Raw value per softplus (0 = 1x)

        # --- LOGICA BUY ---
        # Condizione: Il prezzo sale > 2.5% E non scende sotto -1.5% PRIMA di salire
        sl_threshold_price = current_price * (1 - MAX_STOP_LOSS_TOLERANCE) # SL ipotetico fisso per il check
        tp_threshold_price = current_price * (1 + MIN_PROFIT_PCT)

        # Indici temporali (quando succede cosa)
        try:
            idx_tp_hit = next(i for i, x in enumerate(highs) if x > tp_threshold_price)
            idx_sl_hit = next(i for i, x in enumerate(lows) if x < sl_threshold_price)
        except StopIteration:
            # Se non trova TP o SL, metti indici infiniti
            idx_tp_hit = 9999 if max_up_pct <= MIN_PROFIT_PCT else 999
            idx_sl_hit = 9999 if max_down_pct <= 0.015 else 999

        if idx_tp_hit < idx_sl_hit:
            # === BUY SIGNAL ===
            target_side = 0
            target_qty = 0.95 # Usa 95% budget

            # CALCOLO TP IDEALE
            # Il TP ideale è il massimo raggiunto (un po' meno per sicurezza, es. 90% del movimento)
            # Decoder Logic: tp_price = limit_price * (1 + tp_mult * 0.10)
            # Quindi: tp_mult = (pct_gain / 0.10)
            best_exit_price = max(highs[:idx_tp_hit+5]) # Cerca max locale attorno al TP
            pct_gain = (best_exit_price - current_price) / current_price
            target_tp_mult = pct_gain / 0.10

            # CALCOLO SL IDEALE
            # Lo SL deve essere sotto il minimo toccato PRIMA del TP
            lowest_before_tp = min(lows[:idx_tp_hit+1])
            # Aggiungiamo un buffer di sicurezza (0.2%)
            safe_sl_price = lowest_before_tp * 0.998
            pct_loss = (current_price - safe_sl_price) / current_price
            # Decoder Logic: sl_price = limit_price * (1 - sl_mult * 0.05)
            # Quindi: sl_mult = (pct_loss / 0.05)
            target_sl_mult = pct_loss / 0.05

            # ORDER TYPE
            # Se nella prima ora (candela 0) fa già +1%, entra MARKET!
            if (closes[0] - current_price)/current_price > 0.01:
                target_ordertype = 1 # Market

        # --- LOGICA SELL ---
        elif max_down_pct > MIN_PROFIT_PCT: # Semplificato per ora
            # Check speculare (omesso per brevità, assumiamo SELL se scende forte)
            # In un bot Spot only, SELL significa "Vendi se hai crypto".
            # Ma qui assumiamo logica Trading (Short).
            target_side = 1
            target_qty = 0.95

            # TP (Short)
            best_low = min(lows)
            pct_gain = (current_price - best_low) / current_price
            target_tp_mult = pct_gain / 0.10

            # SL (Short)
            highest_high = max(highs) # Semplificato
            pct_loss = (highest_high - current_price) / current_price
            target_sl_mult = pct_loss / 0.05

        # --- CLAMPING VALORI ---
        # Evitiamo valori assurdi che fanno esplodere la loss
        target_tp_mult = max(0.1, min(target_tp_mult, 5.0)) # Minimo un po' di TP, max 50%
        target_sl_mult = max(0.1, min(target_sl_mult, 5.0)) # Minimo un po' di SL, max 25%

        return {
            "side": torch.tensor([target_side], dtype=torch.long),
            "qty": torch.tensor([target_qty], dtype=torch.float32).view(-1, 1),
            "px_offset": torch.tensor([0.0], dtype=torch.float32).view(-1, 1), # Market entry
            "tp_mult": torch.tensor([target_tp_mult], dtype=torch.float32).view(-1, 1),
            "sl_mult": torch.tensor([target_sl_mult], dtype=torch.float32).view(-1, 1),
            "ordertype": torch.tensor([target_ordertype], dtype=torch.long),
            "leverage": torch.tensor([0.0], dtype=torch.float32).view(-1, 1), # 0.0 raw = 1x dopo softplus
            "halt": torch.tensor([1.0], dtype=torch.float32).view(-1, 1) # Target Confidence
        }

    def train_step(self, context, pair_limits, future_candles):
        self.model.train()
        self.optimizer.zero_grad()

        # 1. Simulazione Wallet
        if random.random() < 0.2:
            simulated_wallet = random.uniform(0.1, 9.0)
        else:
            simulated_wallet = random.uniform(20.0, 5000.0)

        inputs, ref_price = self.vectorizer.vectorize(
            candles_db_data=context['candles'],
            open_order=None,
            forecast_db_data=context['forecast'],
            pair_limits=pair_limits,
            wallet_balance=simulated_wallet
        )

        device = next(self.model.parameters()).device
        for k, v in inputs.items():
            inputs[k] = v.to(device)

        # 2. Forward Pass
        y, _ = self.model(inputs, h=None)
        preds = self.model.get_heads_dict(y)

        # 3. Oracle Labeling
        ordermin = 10.0 # Valore fisso sicuro
        current_close = context['candles']['1h'][0]['close']

        targets = self.generate_oracle_label(future_candles, current_close, simulated_wallet, ordermin)

        if targets is None:
            return None

        # Spostiamo target su device
        t_side = targets['side'].to(device)
        t_qty = targets['qty'].to(device)
        t_px = targets['px_offset'].to(device)
        t_tp = targets['tp_mult'].to(device)
        t_sl = targets['sl_mult'].to(device)
        t_type = targets['ordertype'].to(device)
        t_lev = targets['leverage'].to(device)
        t_halt = targets['halt'].to(device)

        # --- 4. CALCOLO LOSS COMPLETA ---

        # A. Decisione Base (Side) - Priorità Alta
        loss_side = self.loss_ce(preds['side'], t_side)

        # Mask: Le loss numeriche contano SOLO se non è HOLD
        # Se devo fare HOLD, non mi importa che TP/SL predice il modello
        is_active = (t_side != 2).float().view(-1, 1)

        # B. Parametri Ordine (Qty, TP, SL, Price, Lev)
        loss_qty = self.loss_mse(preds['qty'], t_qty) * is_active
        loss_tp  = self.loss_mse(preds['tp_mult'], t_tp) * is_active
        loss_sl  = self.loss_mse(preds['sl_mult'], t_sl) * is_active
        loss_px  = self.loss_mse(preds['price_offset'], t_px) * is_active
        loss_lev = self.loss_mse(preds['leverage'], t_lev) * is_active # Allena a stare a 1x per ora

        # C. Order Type (Limit vs Market)
        # Nota: preds['ordertype'] è logits [Batch, 2], t_type è [Batch]
        loss_type = self.loss_ce(preds['ordertype'], t_type) * is_active.view(-1)

        # D. Halt (Confidence)
        loss_halt = self.loss_bce(preds['halt_prob'], t_halt)

        # --- 5. SOMMA PESATA ---
        # Diamo pesi diversi in base all'importanza
        total_loss = (
            2.0 * loss_side +   # Fondamentale azzeccare la direzione
            1.0 * loss_qty +    # Importante non sbagliare size
            0.5 * loss_tp +     # Importante per profitto
            0.5 * loss_sl +     # Importante per sicurezza
            0.2 * loss_type +   # Meno importante
            0.2 * loss_px +     # Meno importante (per ora targettiamo market)
            0.1 * loss_lev +    # Basso peso, tanto il target è fisso
            0.2 * loss_halt     # Deve imparare a essere deciso
        )

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return {
            "loss": total_loss.item(),
            "target_side": t_side.item(),
            "pred_side": torch.argmax(preds['side']).item()
        }

    def save_checkpoint(self, path="model_checkpoint.pth"):
        torch.save(self.model.state_dict(), path)
        print(f"--- Modello salvato in {path} ---")
