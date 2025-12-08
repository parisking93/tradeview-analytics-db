import torch
import torch.nn.functional as F
import sys
import os
import json # Serve per parsare l'array dal DB

class ActionDecoder:
    def __init__(self, ref_price: float, pair: dict, max_qty: float = 50.0, order: dict = None):

        self.ref_price = ref_price
        self.pair_name = pair.get('pair')
        self.pair = pair
        self.max_qty = max_qty

        # Estrazione limiti dal DB (con fallback sicuri)
        limits = pair.get('pair_limits') or {}
        self.pair_decimals = int(limits.get('pair_decimals', 2))
        self.lot_decimals = int(limits.get('lot_decimals', 8))
        self.min_order_qty = float(limits.get('ordermin', 0.0))

        # Info Leva
        self.lev_buy_limits = limits.get('leverage_buy')
        self.lev_sell_limits = limits.get('leverage_sell')
        self.order = order

    def decode(self, heads: dict, step_k: int) -> dict:
        """
        Decodifica i tensori in un ordine valido per l'Exchange.
        """
        # --- 1. SIDE & CONFIDENCE ---
        side_logits = heads['side'][0]
        probs = F.softmax(side_logits, dim=0)
        side_idx = torch.argmax(probs).item()
        side_map = {0: "BUY", 1: "SELL", 2: "HOLD"}
        decision = side_map[side_idx]
        confidence = probs[side_idx].item()

        # --- 2. ORDER TYPE ---
        # 0 = LIMIT, 1 = MARKET
        ordertype_idx = torch.argmax(heads['ordertype'][0]).item()
        ordertype = "LIMIT" if ordertype_idx == 0 else "MARKET"

        # --- 3. PREZZO (Limit vs Market) ---
        # Il modello suggerisce un offset (es. -0.01 = 1% sotto).
        px_offset = heads['price_offset'].item() * 0.05 # Range max +/- 5%

        if ordertype == "MARKET":
            # Se Market, usiamo il prezzo di riferimento attuale senza modifiche
            limit_price = self.ref_price
        else:
            # Se Limit, applichiamo l'offset al prezzo di riferimento
            # Nota: offset negativo = prezzo più basso (buono per Buy Limit)
            # offset positivo = prezzo più alto (buono per Sell Limit)
            limit_price = self.ref_price * (1.0 + px_offset)

        # Arrotondamento obbligatorio
        limit_price = round(limit_price, self.pair_decimals)

        # --- 4. QUANTITÀ (EUR -> Units -> Min/Max Range) ---
        qty_model_pct = heads['qty'].item() # Valore tra 0.0 e 1.0 (Sigmoid)

        # A. Convertiamo il budget EUR in unità della moneta (es. ETH)
        # Usiamo ref_price per la stima
        max_qty_units = self.max_qty / (self.ref_price + 1e-9)

        # B. Calcolo Quantità Reale
        # La quantità deve stare tra min_order_qty e max_qty_units
        if max_qty_units < self.min_order_qty:
            # Se il budget non basta nemmeno per il minimo ordine, forziamo 0
            final_qty = 0.0
            if decision != "HOLD": decision = "HOLD (No Budget)"
        else:
            # Interpolazione lineare: Min + (Range * Percentuale_Modello)
            qty_range = max_qty_units - self.min_order_qty
            raw_qty = self.min_order_qty + (qty_range * qty_model_pct)

            # Arrotondamento obbligatorio
            final_qty = round(raw_qty, self.lot_decimals)

        # --- 5. LEVA (Snap-to-Grid) ---
        raw_lev = heads['leverage'].item()
        is_buy = (decision == "BUY")
        final_leverage = self._snap_leverage(raw_lev, is_buy)

        # --- 6. TAKE PROFIT & STOP LOSS ---
        # Calcolati a partire dal prezzo di esecuzione (limit_price)
        tp_dist = heads['tp_mult'].item() * 0.10 # Max 10%
        sl_dist = heads['sl_mult'].item() * 0.05 # Max 5%

        take_profit = 0.0
        stop_loss = 0.0

        if decision == "BUY":
            take_profit = limit_price * (1 + tp_dist)
            stop_loss = limit_price * (1 - sl_dist)
        elif decision == "SELL":
            take_profit = limit_price * (1 - tp_dist)
            stop_loss = limit_price * (1 + sl_dist)

        # Arrotondamento TP/SL
        take_profit = round(take_profit, self.pair_decimals)
        stop_loss = round(stop_loss, self.pair_decimals)


        return {
            "step": step_k,
            "pair": self.pair_name,
            "decision": decision,
            "confidence": confidence,
            "limit_price": limit_price,
            "qty_percent": qty_model_pct, # Info debug
            "final_qty": final_qty,              # Qty effettiva per l'ordine
            "take_profit": take_profit,
            "stop_loss": stop_loss,
            "ordertype": ordertype,
            "leverage": final_leverage,
            "actionKraken": {
                "pair": self.pair_name,
                "tipo": decision,
                "ordertype": ordertype,
                "quando": ordertype,
                "prezzo": limit_price,
                "quantita": final_qty,
                "quantita_eur": round(final_qty * limit_price, 2),
                "leverage": final_leverage,
                "take_profit": take_profit,
                "stop_loss": stop_loss,
                "timeframe": "24H",
                "lato": decision,
                "limite": limit_price if ordertype == "LIMIT" else None,
                "reduce_only": False
            }
        }

    def print_action(self, action: dict, is_final: bool = False):
        prefix = "[FINAL DECISION]" if is_final else f"[Thinking Step {action['step']}]"
        color = "\033[92m" if action['decision'] == "BUY" else "\033[91m" if action['decision'] == "SELL" else "\033[93m"
        reset = "\033[0m"

        print(f"{prefix} {self.pair_name} -> {color}{action['decision']} ({action['confidence']:.1%}){reset}")

        if action['decision'] != "HOLD":
            print(f"   Price: {action['limit_price']:.4f} | Qty: {action['qty_percent']:.1%} size | TP: {action['take_profit']:.4f} | SL: {action['stop_loss']:.4f} |  Lev: {action['leverage']:.4f} | OrderType: {action['ordertype']}")


    def _snap_leverage(self, raw_lev: float, is_buy: bool) -> float:
        """
        Prende il valore grezzo del modello (es. 3.45) e lo aggancia
        al valore valido più vicino nel database (es. 3).
        """
        # 1. Recupera la lista valida dal DB (è salvata come stringa JSON '[2, 3, 4]')
        try:
            key = 'leverage_buy' if is_buy else 'leverage_sell'
            allowed_json = (self.pair.get('pair_limits') or {}).get(key)

            if not allowed_json:
                return 1.0 # Nessuna leva permessa -> Spot

            # Se è già lista ok, se è stringa parsa
            if isinstance(allowed_json, str):
                allowed_levels = json.loads(allowed_json)
            else:
                allowed_levels = allowed_json

            if not allowed_levels:
                return 1.0

        except Exception as e:
            print(f"[ERR] Errore parsing leva: {e}")
            return 1.0

        # 2. Se il modello suggerisce < 1.5, intende probabilmente 1x (Spot)
        # A meno che 1x non sia esplicitamente vietato (raro)
        if raw_lev < 1.5:
            return 1.0

        # 3. Trova il valore più vicino nella lista
        # Es: raw=3.8, levels=[2, 3, 4, 5] -> vince 4
        # key=lambda x: abs(x - raw_lev) calcola la distanza
        allowed_levels = [1] + allowed_levels  # Aggiungi sempre 1x come opzione
        best_lev = min([1] + allowed_levels, key=lambda x: abs(x - raw_lev))

        return float(best_lev)
