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

        # Optimizer: AdamW è lo standard de-facto per i Transformer/RNN moderni
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)

        # Loss Functions
        self.loss_side = nn.CrossEntropyLoss()  # Per Buy/Sell/Hold (Classificazione)
        self.loss_value = nn.MSELoss()          # Per numeri continui (Prezzo, Qty, TP/SL)

    def generate_oracle_label(self, future_candles, current_price):
        """
        Guarda nel futuro e decide quale SAREBBE STATA l'azione migliore.
        Ritorna un dizionario con i target ideali.
        """
        if not future_candles or len(future_candles) < 5:
            return None # Dati insufficienti

        # Parametri dell'Oracolo (Cosa consideriamo un "buon trade"?)
        MIN_PROFIT_PCT = 0.030  # 5% profitto minimo richiesto
        MAX_STOP_LOSS = 0.015   # 1.5% stop loss massimo tollerato

        highs = [c['high'] for c in future_candles]
        lows = [c['low'] for c in future_candles]
        closes = [c['close'] for c in future_candles]

        max_high = max(highs)
        min_low = min(lows)

        # Calcolo variazioni massime
        max_up_pct = (max_high - current_price) / current_price
        max_down_pct = (current_price - min_low) / current_price

        # --- LOGICA DELL'ORACOLO ---
        target_side = 2 # Default: HOLD
        target_px_offset = 0.0

        # 1. Analisi LONG
        # Se sale più del target, e nel frattempo non scende sotto lo stop loss
        first_sl_hit_index = next((i for i, x in enumerate(lows) if x < current_price * (1 - MAX_STOP_LOSS)), 9999)
        first_tp_hit_index = next((i for i, x in enumerate(highs) if x > current_price * (1 + MIN_PROFIT_PCT)), 9999)

        if max_up_pct > MIN_PROFIT_PCT and first_tp_hit_index < first_sl_hit_index:
            target_side = 0 # BUY
            # Target Price: Puntiamo a prendere un po' meno del massimo assoluto (più realistico)
            ideal_entry = current_price # Per ora assumiamo Market entry
            target_px_offset = 0.0      # 0% offset (Market)

        # 2. Analisi SHORT (se abilitato, per ora logica semplice)
        elif max_down_pct > MIN_PROFIT_PCT:
            # Qui servirebbe controllo indice simile al Long
            # Per semplicità in V1, se scende molto è SELL
            target_side = 1 # SELL
            target_px_offset = 0.0

        # --- Creazione Tensori Target ---
        return {
            "side": torch.tensor([target_side], dtype=torch.long), # 0, 1, o 2
            "px_offset": torch.tensor([target_px_offset], dtype=torch.float32),
            "qty": torch.tensor([0.9 if target_side != 2 else 0.0], dtype=torch.float32) # Se trade, usa 90% budget
        }

    def train_step(self, context, pair_limits, future_candles):
        """
        Esegue un singolo passo di training su un datapoint.
        """
        self.model.train() # Modalità Training (attiva Dropout, ecc.)
        self.optimizer.zero_grad() # Reset gradienti

        # 1. Vettorializzazione (SIMULANDO un Wallet Balance random per robustezza)
        # Randomizziamo il wallet tra 100 e 10.000 per insegnare al modello a gestire budget diversi
        simulated_wallet = random.uniform(100.0, 10000.0)

        inputs, ref_price = self.vectorizer.vectorize(
            candles_db_data=context['candles'],
            open_order=None, # In training assumiamo di partire flat
            forecast_db_data=context['forecast'],
            pair_limits=pair_limits,
            wallet_balance=simulated_wallet
        )

        # Spostiamo su GPU se disponibile
        device = next(self.model.parameters()).device
        for k, v in inputs.items():
            inputs[k] = v.to(device)

        # 2. Forward Pass (Cosa pensa il modello?)
        # Passiamo h=None perché stiamo allenando su campioni random, non sequenze continue
        y, _ = self.model(inputs, h=None)
        preds = self.model.get_heads_dict(y)

        # 3. Generazione Label (Cosa avrebbe dovuto fare?)
        # Usiamo il prezzo di chiusura dell'ultima candela 1h nota come riferimento corrente
        current_close = context['candles']['1h'][0]['close']
        targets = self.generate_oracle_label(future_candles, current_close)

        if targets is None:
            return None, 0.0 # Skip step

        # Spostiamo target su device
        t_side = targets['side'].to(device)
        t_px = targets['px_offset'].to(device)
        t_qty = targets['qty'].to(device)

        # 4. Calcolo Loss (Errore)
        # Loss Totale = Loss Decisione + Loss Prezzo + Loss Quantità
        loss_s = self.loss_side(preds['side'], t_side)

        # Calcoliamo la loss sui valori numerici SOLO se l'azione non era HOLD
        # Altrimenti il modello impazzisce cercando di predire prezzi per un HOLD
        mask_active = (t_side != 2).float()

        loss_p = self.loss_value(preds['price_offset'], t_px) * mask_active
        loss_q = self.loss_value(preds['qty'], t_qty) * mask_active

        # Somma pesata (Side è la più importante)
        total_loss = 2.0 * loss_s + 0.5 * loss_p + 0.5 * loss_q

        # 5. Backpropagation (Imparare)
        total_loss.backward()

        # Gradient Clipping (evita esplosioni numeriche nei network ricorrenti)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        return {
            "loss": total_loss.item(),
            "acc_side": (torch.argmax(preds['side']) == t_side).float().item(),
            "pred_side": torch.argmax(preds['side']).item(),
            "target_side": t_side.item()
        }

    def save_checkpoint(self, path="model_checkpoint.pth"):
        torch.save(self.model.state_dict(), path)
        print(f"--- Modello salvato in {path} ---")
