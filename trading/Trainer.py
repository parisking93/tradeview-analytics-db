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

        # Carica pesi (Best effort)
        try:
        #     self.model.load_state_dict(torch.load("trm_model_v2.pth"), strict=False)
            self.model.load_state_dict(torch.load("trm_model_best.pth"), strict=False)
            print("--- Pesi 'Best Model' caricati ---")
        except:
            print("--- Nessun peso precedente, start fresh ---")

        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)

        # --- SCHEDULER ---
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=500
        )
        self.accumulation_steps = 4
        # --- LOSS FUNCTIONS SEPARATE ---

        # 1. Loss per il SIDE (3 classi: Buy, Sell, Hold)
        # Qui USIAMO i pesi per bilanciare l'Hold
        self.weights_side = torch.tensor([2.0, 2.0, 1.0])
        self.loss_ce_side = nn.CrossEntropyLoss(weight=self.weights_side)

        # 2. Loss per ORDER TYPE (2 classi: Limit, Market)
        # Qui NON usiamo pesi (o standard), perché Limit e Market sono bilanciati
        self.loss_ce_type = nn.CrossEntropyLoss()

        self.loss_mse = nn.MSELoss()
        self.loss_bce = nn.BCELoss()

    def generate_oracle_label(self, future_candles, current_price, wallet_balance, min_order_cost=10.0, pair_limits=None):
        """
        Calcola i target ideali.
        """
        if not future_candles or len(future_candles) < 5:
            return None

        # --- 1. CHECK POVERTÀ ---
        if wallet_balance < min_order_cost:
            return {
                "side": torch.tensor([2], dtype=torch.long),
                "qty": torch.tensor([0.0], dtype=torch.float32).view(-1, 1),
                "px_offset": torch.tensor([0.0], dtype=torch.float32).view(-1, 1),
                "tp_mult": torch.tensor([0.0], dtype=torch.float32).view(-1, 1),
                "sl_mult": torch.tensor([0.0], dtype=torch.float32).view(-1, 1),
                "ordertype": torch.tensor([0], dtype=torch.long),
                "leverage": torch.tensor([0.0], dtype=torch.float32).view(-1, 1),
                "halt_prob": torch.tensor([1.0], dtype=torch.float32).view(-1, 1)
            }

        # --- 2. ANALISI MERCATO ---
        MIN_PROFIT_PCT = 0.025
        MAX_STOP_LOSS_TOLERANCE = 0.020
        RISK_PER_TRADE = 0.01

        highs = [c['high'] for c in future_candles]
        lows = [c['low'] for c in future_candles]
        closes = [c['close'] for c in future_candles]

        max_high = max(highs)
        min_low = min(lows)
        max_up_pct = (max_high - current_price) / current_price
        max_down_pct = (current_price - min_low) / current_price

        # Default: HOLD
        target_side = 2
        target_qty = 0.0
        target_tp_mult = 0.0
        target_sl_mult = 0.0
        target_ordertype = 0
        target_leverage = 0.0

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

            best_exit_price = max(highs[:idx_tp_hit+5])
            pct_gain = (best_exit_price - current_price) / current_price
            target_tp_mult = pct_gain / 0.10

            lowest_before_tp = min(lows[:idx_tp_hit+1])
            safe_sl_price = lowest_before_tp * 0.998
            pct_loss = (current_price - safe_sl_price) / current_price
            pct_loss = max(0.002, pct_loss)
            target_sl_mult = pct_loss / 0.05

            raw_lev = RISK_PER_TRADE / pct_loss
            max_pair_lev = float(pair_limits.get('leverage_buy_max', 1.0)) if pair_limits else 1.0
            safe_lev = min(raw_lev, max_pair_lev, 5.0)
            target_leverage = max(0.0, safe_lev - 1.0)

            if (closes[0] - current_price)/current_price > 0.008:
                target_ordertype = 1
            else:
                target_ordertype = 0

        # --- LOGICA SELL ---
        elif max_down_pct > MIN_PROFIT_PCT:
            target_side = 1
            target_qty = 0.95

            best_low = min(lows)
            pct_gain = (current_price - best_low) / current_price
            target_tp_mult = pct_gain / 0.10

            highest_before_low = max(highs)
            pct_loss = (highest_before_low - current_price) / current_price
            pct_loss = max(0.002, pct_loss)
            target_sl_mult = pct_loss / 0.05

            raw_lev = RISK_PER_TRADE / pct_loss
            max_pair_lev = float(pair_limits.get('leverage_sell_max', 1.0)) if pair_limits else 1.0
            safe_lev = min(raw_lev, max_pair_lev, 5.0)
            target_leverage = max(0.0, safe_lev - 1.0)

        target_tp_mult = max(0.1, min(target_tp_mult, 5.0))
        target_sl_mult = max(0.1, min(target_sl_mult, 5.0))

        return {
            "side": torch.tensor([target_side], dtype=torch.long),
            "qty": torch.tensor([target_qty], dtype=torch.float32).view(-1, 1),
            "px_offset": torch.tensor([0.0], dtype=torch.float32).view(-1, 1),
            "tp_mult": torch.tensor([target_tp_mult], dtype=torch.float32).view(-1, 1),
            "sl_mult": torch.tensor([target_sl_mult], dtype=torch.float32).view(-1, 1),
            "ordertype": torch.tensor([target_ordertype], dtype=torch.long),
            "leverage": torch.tensor([target_leverage], dtype=torch.float32).view(-1, 1),
            "halt_prob": torch.tensor([1.0], dtype=torch.float32).view(-1, 1)
        }

    def train_step(self, context, pair_limits, future_candles, current_step_idx):
        self.model.train()
        # NOTA: Non azzeriamo i gradienti qui! Lo facciamo solo dopo l'accumulo.


        # 1. Simulazione Wallet (Uguale)
        if random.random() < 0.2: simulated_wallet = random.uniform(0.1, 9.0)
        else: simulated_wallet = random.uniform(20.0, 5000.0)

        inputs, ref_price = self.vectorizer.vectorize(
            candles_db_data=context['candles'],
            open_order=None,
            forecast_db_data=context['forecast'],
            pair_limits=pair_limits,
            wallet_balance=simulated_wallet
        )

        device = next(self.model.parameters()).device
        # Assegniamo i pesi solo alla loss del Side
        if self.loss_ce_side.weight.device != device:
             self.loss_ce_side.weight = self.weights_side.to(device)

        for k, v in inputs.items(): inputs[k] = v.to(device)

        # 2. Forward
        y, _ = self.model(inputs, h=None)
        preds = self.model.get_heads_dict(y)

        # 3. Labeling
        current_close = context['candles']['1h'][0]['close']
        targets = self.generate_oracle_label(future_candles, current_close, simulated_wallet, 10.0, pair_limits)

        if targets is None: return None

        t_side = targets['side'].to(device)
        t_qty = targets['qty'].to(device)
        t_px = targets['px_offset'].to(device)
        t_tp = targets['tp_mult'].to(device)
        t_sl = targets['sl_mult'].to(device)
        t_type = targets['ordertype'].to(device)
        t_lev = targets['leverage'].to(device)
        t_halt = targets['halt_prob'].to(device)

        # 4. Loss Calcolo
        is_active = (t_side != 2).float().view(-1, 1)

        loss_side = self.loss_ce_side(preds['side'], t_side)
        loss_qty = self.loss_mse(preds['qty'], t_qty) * is_active
        loss_tp  = self.loss_mse(preds['tp_mult'], t_tp) * is_active
        loss_sl  = self.loss_mse(preds['sl_mult'], t_sl) * is_active
        loss_px  = self.loss_mse(preds['price_offset'], t_px) * is_active
        loss_lev = self.loss_mse(preds['leverage'], t_lev) * is_active
        loss_type = self.loss_ce_type(preds['ordertype'], t_type) * is_active.view(-1)
        loss_halt = self.loss_bce(preds['halt_prob'], t_halt)

        total_loss = (
            3.0 * loss_side +
            1.0 * loss_qty +
            0.5 * loss_tp +
            0.5 * loss_sl +
            0.3 * loss_lev +
            0.2 * loss_type +
            0.1 * loss_px +
            0.2 * loss_halt
        )

        # --- GESTIONE GRADIENT ACCUMULATION ---

        # Normalizziamo la loss (perché sommeremo i gradienti 32 volte)
        loss_normalized = total_loss / self.accumulation_steps
        loss_normalized.backward() # Accumula il gradiente

        # Se abbiamo raggiunto il numero di step o siamo alla fine, aggiorniamo
        if (current_step_idx + 1) % self.accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.optimizer.zero_grad() # Resetta ORA, dopo l'update

            # Aggiorna scheduler (opzionale farlo qui o a fine epoca)
            # self.scheduler.step(total_loss)

        return {
            "loss": total_loss.item(), # Ritorniamo la loss vera (non normalizzata) per i log
            "target_side": t_side.item(),
            "pred_side": torch.argmax(preds['side']).item()
        }

    def save_checkpoint(self, path="model_checkpoint.pth"):
        torch.save(self.model.state_dict(), path)
        # print(f"--- Saved {path} ---")
