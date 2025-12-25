import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
import zlib
from datetime import timedelta, datetime

class TradingTrainer:
    def __init__(self, model, db_manager, vectorizer, learning_rate=2e-5):
        self.model = model
        self.db = db_manager
        self.vectorizer = vectorizer

        # --- HALT tuning ---
        self.halt_gamma = 2.0
        self.halt_alpha_pos = 0.6  # Meno aggressivo su HOLD
        self.halt_alpha_neg = 0.4  # Più peso ai trade quando sono reali
        self.halt_loss_weight = 0.6 # Aumentato peso per forzare il modello a decidere meglio se stare fermo

        # --- SIDE Focal Tuning ---
        self.side_gamma = 2.0
        self.side_weights = torch.tensor([1.2, 1.2, 1.0]) # Ribilanciato: meno aggressivo sui trade per evitare over-trading


        # Carica pesi (Best effort)
        try:
        #     self.model.load_state_dict(torch.load("trm_model_v2.pth"), strict=False)
            self.model.load_state_dict(torch.load("model/trainerLast.pth"), strict=False)
            print("--- Pesi 'Best Model' caricati ---")
        except:
            print("--- Nessun peso precedente, start fresh ---")

        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)

        # --- SCHEDULER ---
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=50
        )
        self.accumulation_steps = 8 # Aumentato per gradienti più stabili (batch virtuale più grande)

        # --- MULTISTEP THINKING ---
        # Numero di step di "pensiero" durante il training (unroll della GRUCell del cervello)
        self.thinking_steps = 5  # Ridotto da 7 per stabilità gradienti

        # --- LOSS FUNCTIONS SEPARATE ---

        # 1. Loss per il SIDE (Focal Loss custom in train_step)
        self.loss_ce_side_raw = nn.CrossEntropyLoss(reduction='none')
        self.loss_lev = torch.nn.SmoothL1Loss(reduction='none')

        # 2. Loss per ORDER TYPE (2 classi: Limit, Market)
        # Qui NON usiamo pesi (o standard), perche Limit e Market sono bilanciati
        self.loss_ce_type = nn.CrossEntropyLoss()

        self.loss_mse = nn.MSELoss()
        self.loss_bce = nn.BCELoss()
        # Counters to compute dynamic class weights for ordertype (0=LIMIT,1=MARKET)
        # Initialized to 1 to avoid division by zero
        self.type_counts = [1.0, 1.0]

    def generate_fake_order(self, context, pair_limits):
        """
        Genera un ordine finto basato su candele di 1-2 giorni fa.
        Popola l'oggetto ordine come se venisse dal DB.
        """
        # Se c'è già un ordine reale (o finto generato in precedenza), usciamo
        if context.get('order') is not None:
            return None

        # Parametro di casualità: 40% di probabilità di avere un ordine aperto
        if random.random() > 0.4:
            return None

        # Recuperiamo le candele 1h
        candles = context['candles'].get('5m', [])
        candles1h = context['candles'].get('1h', [])
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
        target_start = ts_val - timedelta(hours=30)
        target_end = ts_val - timedelta(hours=16)

        # Filtriamo le candele candidate
        candidates = []
        for c in candles1h:
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

        # --- NUOVO: Decidere orderType ---
        # 30-40% di probabilità che sia un LIMIT order (pendente), altrimenti MARKET (eseguito subito)
        is_limit = random.random() < 0.35
        order_type_str = "LIMIT" if is_limit else "MARKET"

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
            "orderType": order_type_str,  # NUOVO: LIMIT o MARKET
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
                order_type_str = fake_order.get('orderType', 'MARKET')  # NUOVO: LIMIT o MARKET

                # --- NUOVO: Gestione LIMIT PENDING ---
                # Se l'ordine è LIMIT e status è OPEN, il target DEVE essere HOLD
                # (non vogliamo che il modello replazzi continuamente ordini LIMIT pendenti)
                if order_type_str == "LIMIT" and fake_order.get('status') == 'OPEN':
                    return {
                        "side": torch.tensor([2], dtype=torch.long),  # HOLD
                        "qty": torch.tensor([0.0], dtype=torch.float32).view(-1, 1),
                        "px_offset": torch.tensor([0.0], dtype=torch.float32).view(-1, 1),
                        "tp_mult": torch.tensor([0.0], dtype=torch.float32).view(-1, 1),
                        "sl_mult": torch.tensor([0.0], dtype=torch.float32).view(-1, 1),
                        "ordertype": torch.tensor([0], dtype=torch.long),  # LIMIT
                        "leverage": torch.tensor([order_lev], dtype=torch.float32).view(-1, 1),
                        "halt_prob": torch.tensor([1.0], dtype=torch.float32).view(-1, 1)  # Massima priorità HOLD
                    }

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

            # --- 2. ANALISI MERCATO (LOGICA AVANZATA) ---
            MIN_PROFIT_PCT = 0.02
            MAX_STOP_LOSS_TOLERANCE = 0.03
            RISK_PER_TRADE = 0.02

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

            # Default: HOLD
            target_side = 2
            target_qty = 0.0
            target_tp_mult = 0.0
            target_sl_mult = 0.0
            target_px_offset = 0.0
            target_ordertype = 1
            target_leverage = 1.0
            target_halt = 0.95

            # Calcola ATR% per soglia dinamica MARKET/LIMIT (si adatta alla volatilità della currency)
            atr_pct_global = _estimate_atr_pct(highs[:8], lows[:8], closes[:8])
            # Soglia: se il prezzo si muove più di ~30% dell'ATR, entriamo MARKET
            market_threshold = 0.3 * atr_pct_global

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

                # Se il prezzo si muove significativamente -> entra market, altrimenti prova limit su supporto
                # Soglia dinamica basata su ATR% della currency
                if (closes[4] - current_price)/current_price > market_threshold:
                    target_ordertype = 1  # MARKET
                    entry_price = float(current_price)
                    target_px_offset = 0.0
                else:
                    target_ordertype = 0  # LIMIT
                    entry_price, target_px_offset = _compute_limit_entry_and_px(
                        float(current_price), 0, highs, lows, closes
                    )

                # OPTIONAL: If the computed LIMIT would never be filled in the next candles,
                # convert target to MARKET to avoid creating unrealistic LIMIT targets.
                try:
                    would_fill = any(float(c['low']) <= entry_price for c in future_candles)
                except Exception:
                    would_fill = True
                if not would_fill:
                    target_ordertype = 1  # MARKET
                    target_px_offset = 0.0
                    entry_price = float(current_price)

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

                # Se il prezzo si muove significativamente -> entra market, altrimenti prova limit su resistenza
                # Soglia dinamica basata su ATR% della currency
                if (current_price - closes[4]) / current_price > market_threshold:
                    target_ordertype = 1  # MARKET
                    entry_price = float(current_price)
                    target_px_offset = 0.0
                else:
                    target_ordertype = 0  # LIMIT
                    entry_price, target_px_offset = _compute_limit_entry_and_px(
                        float(current_price), 1, highs, lows, closes
                    )

                # OPTIONAL: If the computed LIMIT would never be filled in the next candles,
                # convert target to MARKET to avoid creating unrealistic LIMIT targets.
                try:
                    would_fill = any(float(c['high']) >= entry_price for c in future_candles)
                except Exception:
                    would_fill = True
                if not would_fill:
                    target_ordertype = 1  # MARKET
                    target_px_offset = 0.0
                    entry_price = float(current_price)

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

            # Ensure TP/SL targets result in at least MIN_TP/SL when decoded
            # Decoder uses pct = mult * 0.10 for TP, mult * 0.05 for SL
            # MIN_TP_PCT = 0.004 -> min_mult = 0.04
            # MIN_SL_PCT = 0.003 -> min_mult = 0.06
            target_tp_mult = max(0.04, min(target_tp_mult, 5.0))
            target_sl_mult = max(0.06, min(target_sl_mult, 5.0))

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


    def train_step(self, context, pair_limits, future_candles, current_step_idx, wallet = False, wallet_budget = 0):
        self.model.train()

        # --- BACKUP STATO RNG ---
        # IMPORTANTE: Salviamo lo stato RNG globale PRIMA di seedare,
        # poi lo ripristiniamo alla fine per non corrompere random.shuffle() nel training loop.
        self._rng_state_backup = random.getstate()

        # --- DETERMINISTIC SEEDING ---
        # Per ridurre il noise nelle label, usiamo un seed fisso per questo sample.
        try:
            pivot_candle = context['candles']['5m'][-1]
            ts_str = str(pivot_candle.get('timestamp_dt') or pivot_candle.get('timestamp'))
            pair_name = str(pair_limits.get('pair', 'unk'))

            # Seed STABILE cross-run (zlib.adler32 è deterministico, hash() di Python no)
            unique_str = f"{pair_name}_{ts_str}"
            seed_val = zlib.adler32(unique_str.encode("utf-8")) & 0xffffffff
            random.seed(seed_val)
        except Exception:
            pass

        # 0. Generazione Ordini Finti (Augmentation)
        # Se non c'è un ordine, proviamo a generarne uno finto per insegnare al modello a gestire posizioni aperte
        fake_order = self.generate_fake_order(context, pair_limits)
        if fake_order:
            context['order'] = fake_order

        # 1. Simulazione Wallet (Uguale)
        simulated_wallet = 0
        if wallet:
            simulated_wallet = wallet_budget
        else:
            if random.random() < 0.2:
                simulated_wallet = random.uniform(0.1, 9.0)
            else:
                simulated_wallet = random.uniform(20.0, 1000.0)

        inputs, ref_price = self.vectorizer.vectorize(
            candles_db_data=context['candles'],
            open_order=context['order'], # Passiamo l'ordine (reale o finto)
            forecast_db_data=context['forecast'],
            pair_limits=pair_limits,
            wallet_balance=simulated_wallet
        )

        device = next(self.model.parameters()).device
        for k, v in inputs.items():
            inputs[k] = v.to(device)

        # 1. Estrazione Context (Transformer) - Run ONCE
        brain_input = self.model.extract_features(inputs)

        # Assegniamo i pesi focal
        if self.side_weights.device != device:
             self.side_weights = self.side_weights.to(device)

        # 2. Labeling (come prima, targets una volta sola)
        current_close = context['candles']['5m'][-1]['close']
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
        is_limit = (t_type == 0).float().view(-1, 1)  # 0=LIMIT, 1=MARKET (target)

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

            # Forward step ricorrente del cervello centrale
            y, h = self.model.think(brain_input, h)
            preds = self.model.get_heads_dict(y)
            last_preds = preds

            # --- SIDE Focal Loss (per-sample) ---
            ce_side = self.loss_ce_side_raw(preds['side'], t_side) # [B]
            probs_side = torch.softmax(preds['side'], dim=-1)
            # px = probabilità assegnata alla classe target
            px_side = probs_side.gather(1, t_side.view(-1, 1)).view(-1)
            # alpha bilanciato per classe
            alpha_side = self.side_weights.gather(0, t_side)
            # Focal formula
            loss_side_step = (alpha_side * (1.0 - px_side) ** self.side_gamma * ce_side).mean()

            # queste hanno mask is_active -> diventano [B,1], le riduciamo con mean()
            loss_qty_step = (self.loss_mse(preds['qty'], t_qty) * is_active).mean().squeeze()
            loss_tp_step  = (self.loss_mse(preds['tp_mult'], t_tp) * is_active).mean().squeeze()
            loss_sl_step  = (self.loss_mse(preds['sl_mult'], t_sl) * is_active).mean().squeeze()
            # Importante: l'offset di prezzo ha senso SOLO quando il target è LIMIT.
            # Se lo alleni anche su MARKET (dove il target è quasi sempre 0), il modello converge a offset≈0 anche quando sceglie LIMIT.
            loss_px_step  = (self.loss_mse(preds['price_offset'], t_px) * is_active * is_limit).mean().squeeze()

            pred_lev = preds['leverage'].clamp(1.0, 5.0)
            t_lev_c  = t_lev.clamp(1.0, 5.0)
            loss_lev_step = (self.loss_lev(pred_lev, t_lev_c) * is_active).mean().squeeze()  # scalare
            # Dynamic weighted CrossEntropy for ordertype to mitigate class imbalance
            # Update running counts of targets
            try:
                t_type_flat = t_type.view(-1)
                t_type_item = int(t_type_flat[0].item())
                if t_type_item in (0, 1):
                    self.type_counts[t_type_item] += 1.0
            except Exception:
                pass

            # Compute weight: give MARKET (idx=1) weight proportional to n_limit/n_market
            w_market = float(self.type_counts[0] / (self.type_counts[1] + 1e-6))
            w_market = max(0.1, min(w_market, 10.0))
            weight_tensor = torch.tensor([1.0, w_market], device=device)

            loss_type_step = F.cross_entropy(preds['ordertype'], t_type.view(-1), weight=weight_tensor).squeeze()
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
        # 4. Loss totale (Ribilanciata per evitare dominanza MSE)
        total_loss = (
            4.0 * loss_side +  # Aumentato da 3.0
            0.3 * loss_qty +   # Ridotto da 1.0
            0.2 * loss_tp +    # Ridotto da 0.5
            0.2 * loss_sl +    # Ridotto da 0.5
            0.2 * loss_lev +
            0.3 * loss_type +
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
            # NOTA: Lo scheduler viene ora chiamato a fine epoca in RunTrainingGpu.py
            # con la media della loss dell'epoca, NON qui per singolo sample.

        # === CALCOLO DINAMICO DEL WALLET ===
        # Aggiorna simulated_wallet in base alle azioni predette dal modello
        updated_wallet = simulated_wallet

        if last_preds is not None:
            current_price = float(context['candles']['5m'][-1]['close'])
            pred_side = torch.argmax(last_preds['side']).item()  # 0=BUY, 1=SELL, 2=HOLD
            pred_qty = float(last_preds['qty'].clamp(0, 1).item())
            pred_lev = float(last_preds['leverage'].clamp(1, 5).item())

            # CASO A: Ordine aperto (fake_order esiste)
            if fake_order is not None:
                entry_price = float(fake_order['price_entry'])
                entry_qty = float(fake_order['qty'])
                entry_lev = float(fake_order['lev'])
                entry_side = 0 if fake_order['subtype'] == 'buy' else 1

                # Calcolo del PnL della posizione aperta
                if entry_side == 0:  # LONG position
                    pnl = (current_price - entry_price) * entry_qty
                else:  # SHORT position
                    pnl = (entry_price - current_price) * entry_qty

                # Calcolo del margine bloccato (collaterale)
                position_value = entry_price * entry_qty
                margin_blocked = position_value / entry_lev

                # Se il modello intende chiudere la posizione (pred_side opposto)
                if (entry_side == 0 and pred_side == 1) or (entry_side == 1 and pred_side == 0):
                    # CHIUSURA: restituisci il margine + il PnL
                    updated_wallet = simulated_wallet + margin_blocked + pnl

                # Se il modello intende HOLD la posizione aperta
                elif pred_side == 2:
                    # HOLD: nessun cambio, mantieni il wallet
                    updated_wallet = simulated_wallet

            # CASO B: Nessun ordine aperto (fresh entry)
            else:
                if pred_side == 0 or pred_side == 1:  # BUY o SELL
                    if pred_qty > 0.01:  # Solo se qty significativa
                        # APERTURA POSIZIONE: blocca margine
                        # Assumendo notional di 100€ * pred_qty * pred_lev
                        notional = 30.0 * pred_qty * pred_lev
                        margin_required = notional / pred_lev
                        updated_wallet = simulated_wallet - margin_required
                elif pred_side == 2:  # HOLD
                    # Nessun ordine aperto, HOLD = no change
                    updated_wallet = simulated_wallet

        # Clamp del wallet (non può diventare negativo, minimo 0.1)
        updated_wallet = max(0.1, updated_wallet)

        return {
            "loss": total_loss.item(),
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
            "simulated_wallet": updated_wallet
        }

        # --- RIPRISTINO STATO RNG ---
        # Cruciale: ripristiniamo lo stato RNG per non corrompere random.shuffle() nel loop principale
        random.setstate(_rng_state_backup)

        return result

    def save_checkpoint(self, path="model/model_checkpoint.pth"):
        torch.save(self.model.state_dict(), path)
        # print(f"--- Saved {path} ---")
