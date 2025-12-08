import time
import json
import os
import sys
import torch
import traceback
from datetime import datetime, timedelta

# --- IMPORTS DAL PROGETTO ESISTENTE ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from db.DatabaseManager import DatabaseManager
from db.MarketDataProvider import MarketDataProvider
from trading.Vectorizer import DataVectorizer, VectorizerConfig
from trading.TrmAgent import MultiTimeframeTRM
from trading.Decoder import ActionDecoder
from trading.KrakenOrderRunner import KrakenOrderRunner
# ==============================================================================
# CONFIGURAZIONE GLOBALE
# ==============================================================================

# File dei pesi
MODEL_PATH_HIGH = "trm_model_best_512_high.pth"
MODEL_PATH_LOW  = "trm_model_best_512_low.pth"

# Configurazione Timeframe
TF_CONFIG_HIGH = {"1d": 30, "4h": 50, "1h": 100}
TF_CONFIG_LOW  = {"1h": 30, "15m": 50, "5m": 100}

# Parametri del Thinking Loop
THINKING_STEPS = 6
MIN_STEPS = 2
HALT_THRESHOLD = 0.70

STATE_FILE = "dual_brain_state.json"
MODE = os.getenv("TRADING_MODE", "TEST").upper()
GLOBAL_WALLET_BALANCE = None
OPEN_ORDER_ATTEMPTS = {}

# ==============================================================================
# GESTIONE STATO E PERSISTENZA
# ==============================================================================

class StateManager:
    def __init__(self, filepath=STATE_FILE):
        self.filepath = filepath
        self.watchlist = {}
        self.conflicts = {}
        self.load_state()

    def load_state(self):
        """
        Carica lo stato dal disco.
        NOTA: Carica la Watchlist (lungo termine) ma RESETTA i conflitti (breve termine).
        """
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r') as f:
                    data = json.load(f)
                    self.watchlist = data.get("watchlist", {})
                    # FIX: Non carichiamo i vecchi conflitti. Si riparte da zero ad ogni avvio.
                    self.conflicts = {}
                print(f"[STATE] Stato caricato: {len(self.watchlist)} coppie in Watchlist. Conflitti resettati.")
            except Exception as e:
                print(f"[ERROR] Errore caricamento stato: {e}")
                self.watchlist = {}
                self.conflicts = {}
        else:
            self.watchlist = {}
            self.conflicts = {}

    def save_state(self):
        try:
            with open(self.filepath, 'w') as f:
                json.dump({
                    "watchlist": self.watchlist,
                    "conflicts": self.conflicts # Salviamo cmq per debug in real-time
                }, f, indent=4, default=str)
        except Exception as e:
            print(f"[ERROR] Errore salvataggio stato: {e}")

    def update_watchlist(self, pair, decision):
        now_iso = datetime.now().isoformat()
        self.watchlist[pair] = {
            "added_at": now_iso,
            "last_high_decision": decision,
        }
        self.save_state()

    def clean_watchlist(self, open_positions_pairs):
        to_remove = []
        now = datetime.now()
        for pair, info in self.watchlist.items():
            try:
                added_at = datetime.fromisoformat(info["added_at"])
                is_expired = (now - added_at) > timedelta(minutes=60)
            except:
                is_expired = True # Se la data è corrotta, rimuovi

            has_position = pair in open_positions_pairs

            if is_expired and not has_position:
                to_remove.append(pair)

        for p in to_remove:
            del self.watchlist[p]
            # Se rimuoviamo dalla watchlist, rimuoviamo anche eventuali conflitti residui
            if p in self.conflicts:
                del self.conflicts[p]

        if to_remove:
            print(f"[STATE] Pulizia Watchlist: rimosse {len(to_remove)} coppie scadute.")
            self.save_state()

    def get_conflict_count(self, pair):
        return self.conflicts.get(pair, 0)

    def increment_conflict(self, pair):
        self.conflicts[pair] = self.conflicts.get(pair, 0) + 1
        self.save_state()
        return self.conflicts[pair]

    def reset_conflict(self, pair):
        if pair in self.conflicts:
            del self.conflicts[pair]
            self.save_state()

# ==============================================================================
# CLASSE BRAIN
# ==============================================================================

class BrainInstance:
    def __init__(self, name, tf_config, model_path):
        self.name = name
        self.tf_config = tf_config
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.vectorizer_config = VectorizerConfig(candle_history_config=tf_config)
        self.print_prefix = f"[{self.name.upper()}]"

    def ensure_model_loaded(self, input_dim_candle, static_dim):
        if self.model is None:
            print(f"{self.print_prefix} Inizializzazione modello...")
            self.model = MultiTimeframeTRM(
                tf_configs=self.tf_config,
                input_size_per_candle=input_dim_candle,
                static_size=static_dim,
                hidden_dim=512
            ).to(self.device)

            if os.path.exists(self.model_path):
                try:
                    self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                    print(f"{self.print_prefix} Pesi caricati da {self.model_path}")
                except Exception as e:
                    print(f"{self.print_prefix} [WARN] Impossibile caricare pesi: {e}. Uso pesi casuali.")
            else:
                print(f"{self.print_prefix} [WARN] File pesi {self.model_path} non trovato. Uso pesi casuali.")

            self.model.eval()

    def think(self, pair_data, context_data):
        vectorizer = DataVectorizer(self.vectorizer_config)
        try:
            inputs, ref_price = vectorizer.vectorize(
                candles_db_data=context_data['candles'],
                open_order=context_data['order'],
                forecast_db_data=context_data['forecast'],
                pair_limits=pair_data.get('pair_limits'),
                wallet_balance=_ensure_global_wallet_balance(context_data)
            )
        except Exception as e:
            print(f"{self.print_prefix} [ERROR] Vectorization failed for {pair_data['pair']}: {e}")
            return None

        self.ensure_model_loaded(vectorizer.candle_dim, vectorizer.static_total_dim)

        decoder = ActionDecoder(ref_price, pair_data, order=context_data['order'])
        h = None
        final_action = None
        inputs_device = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        with torch.no_grad():
            for k in range(THINKING_STEPS):
                steps_taken = k + 1
                y, h = self.model(inputs_device, h)
                current_heads = self.model.get_heads_dict(y)
                halt_prob = current_heads['halt_prob'].item()

                can_stop = (steps_taken >= MIN_STEPS)
                wants_to_stop = (halt_prob >= HALT_THRESHOLD)
                forced_stop = (steps_taken == THINKING_STEPS)

                temp_action = decoder.decode(current_heads, steps_taken)

                if can_stop and (wants_to_stop or forced_stop):
                    final_action = temp_action
                    break

            if final_action is None:
                final_action = decoder.decode(current_heads, THINKING_STEPS)

        return final_action

# ==============================================================================
# LOGICA DI ESECUZIONE
# ==============================================================================

def _safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        try:
            return float(default)
        except Exception:
            return 0.0


def _extract_decision_id_from_results(results):
    if not results:
        return None
    for res in results:
        if not isinstance(res, dict):
            continue
        echo = res.get("_echo") or {}
        if echo.get("_decision_id"):
            return echo.get("_decision_id")
        if echo.get("order_id"):
            return echo.get("order_id")
    return None

def _ensure_global_wallet_balance(context_data):
    global GLOBAL_WALLET_BALANCE
    if GLOBAL_WALLET_BALANCE is None:
        GLOBAL_WALLET_BALANCE = _safe_float((context_data or {}).get("wallet_balance"), default=0.0)
    return GLOBAL_WALLET_BALANCE

def _update_wallet_on_open(cost):
    global GLOBAL_WALLET_BALANCE
    if cost is None: return
    GLOBAL_WALLET_BALANCE = _safe_float(GLOBAL_WALLET_BALANCE, default=0.0)
    GLOBAL_WALLET_BALANCE -= _safe_float(cost, default=0.0)

def _update_wallet_on_close(amount):
    global GLOBAL_WALLET_BALANCE
    if amount is None: return
    GLOBAL_WALLET_BALANCE = _safe_float(GLOBAL_WALLET_BALANCE, default=0.0)
    GLOBAL_WALLET_BALANCE += _safe_float(amount, default=0.0)

def _update_wallet_on_cancel(amount):
    global GLOBAL_WALLET_BALANCE
    if amount is None: return
    GLOBAL_WALLET_BALANCE = _safe_float(GLOBAL_WALLET_BALANCE, default=0.0)
    GLOBAL_WALLET_BALANCE += _safe_float(amount, default=0.0)

def _increment_blocked_attempt(pair):
    OPEN_ORDER_ATTEMPTS[pair] = OPEN_ORDER_ATTEMPTS.get(pair, 0) + 1
    return OPEN_ORDER_ATTEMPTS[pair]

def _reset_blocked_attempt(pair):
    if pair in OPEN_ORDER_ATTEMPTS:
        del OPEN_ORDER_ATTEMPTS[pair]

def persist_order_to_db(action, context=None, pair_info=None, runner_results=None, mode=MODE):
    """
    Registra l'azione nella tabella orders. Se esiste un ordine aperto nel contesto,
    aggiorna quel record (price_out/pnl/status). Altrimenti inserisce un nuovo record OPEN.
    """
    global GLOBAL_WALLET_BALANCE
    existing_order = (context or {}).get("order")
    action_body = action.get("actionKraken", {}) if isinstance(action, dict) else {}
    _ensure_global_wallet_balance(context or {})

    pair_name = action.get("pair") if isinstance(action, dict) else action_body.get("pair")
    base = (pair_info or {}).get("base") or (existing_order or {}).get("base")
    quote = (pair_info or {}).get("quote") or (existing_order or {}).get("quote")
    kr_pair = (pair_info or {}).get("kr_pair") or (existing_order or {}).get("kr_pair")

    if (not base or not quote) and pair_name and "/" in pair_name:
        parts = pair_name.split("/", 1)
        base = base or parts[0]
        quote = quote or parts[1]

    qty = _safe_float(action.get("final_qty") if isinstance(action, dict) else None,
                      default=(action_body.get("quantita") or 0.0))
    limit_price = _safe_float(action.get("limit_price") if isinstance(action, dict) else None,
                              default=action_body.get("prezzo") or 0.0)
    take_profit = _safe_float(action.get("take_profit") if isinstance(action, dict) else None,
                              default=action_body.get("take_profit") or 0.0)
    stop_loss = _safe_float(action.get("stop_loss") if isinstance(action, dict) else None,
                            default=action_body.get("stop_loss") or 0.0)
    lev_value = _safe_float(action.get("leverage") if isinstance(action, dict) else None,
                            default=action_body.get("leverage") or 1.0)
    has_leverage = lev_value > 1

    decision_str = (action.get("decision") if isinstance(action, dict) else None) or ""
    subtype = "buy" if decision_str.upper() == "BUY" else "sell" if decision_str.upper() == "SELL" else "hold"
    order_type = "position_margin" if has_leverage else "position"
    order_exec_type = (
        (action.get("ordertype") if isinstance(action, dict) else None)
        or (action_body.get("ordertype") if isinstance(action_body, dict) else None)
        or "LIMIT"
    )
    order_exec_type = str(order_exec_type).upper()
    now_dt = datetime.now()
    record_date = now_dt.date()
    mode_value = str(mode or MODE or "TEST").upper()

    wallet_id = (existing_order or {}).get("wallet_id") or 4
    decision_id = (existing_order or {}).get("decision_id") or _extract_decision_id_from_results(runner_results)

    db = None
    try:
        db = DatabaseManager()

        if existing_order and existing_order.get("status", "").upper() == "OPEN" and existing_order.get("id"):
            subtype_use = (existing_order.get("subtype") or subtype or "buy").lower()
            entry_price = _safe_float(existing_order.get("price_entry"), default=limit_price)
            qty_use = _safe_float(existing_order.get("qty"), default=qty)
            price_out = limit_price if limit_price > 0 else _safe_float(existing_order.get("price"), default=entry_price)
            existing_order_type = (existing_order.get("orderType") or "").upper() or "MARKET"
            if existing_order_type == "LIMIT":
                pnl = 0.0
            else:
                pnl = (price_out - entry_price) * qty_use if subtype_use == "buy" else (entry_price - price_out) * qty_use
            value_eur = price_out * qty_use
            lev_final = existing_order.get("lev") or lev_value or 1.0

            update_sql = """
                UPDATE orders
                SET price_out = %s,
                    price = %s,
                    value_eur = %s,
                    pnl = %s,
                    status = 'CLOSED',
                    record_date = %s,
                    lev = %s
                WHERE id = %s
            """
            db.cursor.execute(update_sql, (
                price_out,
                price_out,
                value_eur,
                pnl,
                record_date,
                lev_final,
                existing_order["id"]
            ))
            db.conn.commit()
            cash_back = (entry_price * qty_use) + pnl
            _update_wallet_on_close(cash_back)
            return existing_order["id"]

        price_entry = limit_price
        price_avg = price_entry
        price = price_entry
        value_eur = qty * price
        pnl = 0.0
        lev_final = lev_value if lev_value > 0 else 1.0

        insert_sql = """
            INSERT INTO orders (
                wallet_id, pair, kr_pair, base, quote, qty,
                price_entry, price_avg, take_profit, stop_loss,
                price, value_eur, pnl, type, subtype,
                created_at, record_date, status, price_out, decision_id, lev,
                mode, orderType
            ) VALUES (
                %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s,
                %s, %s
            )
        """
        db.cursor.execute(insert_sql, (
            wallet_id, pair_name, kr_pair, base, quote, qty,
            price_entry, price_avg, take_profit, stop_loss,
            price, value_eur, pnl, order_type, subtype,
            now_dt, record_date, "OPEN", None, decision_id, lev_final,
            mode_value, order_exec_type
        ))
        db.conn.commit()
        _update_wallet_on_open(value_eur)
        return db.cursor.lastrowid
    except Exception as e:
        if db and db.conn:
            db.conn.rollback()
        print(f"[ERROR] Persistenza ordine fallita: {e}")
        return None
    finally:
        if db:
            db.close_connection()


def execute_order(action, context=None, pair_info=None, source="LowTF"):
    decision = action.get("decision") if isinstance(action, dict) else getattr(action, "decision", "")
    action_body = action.get("actionKraken") if isinstance(action, dict) else getattr(action, "actionKraken", None)
    if not action_body:
        print("[WARN] Nessun action body disponibile per l'esecuzione.")
        return

    runner = KrakenOrderRunner()
    bodies = runner.build_bodies([action_body], validate=True, auto_brackets=False)
    execution_results = runner.execute_bodies(bodies, timeout=0.8)
    has_success = any(not (res.get("error") or []) for res in execution_results) if execution_results else False
    if has_success:
        try:
            persist_order_to_db(action, context=context, pair_info=pair_info, runner_results=execution_results, mode=MODE)
        except Exception as e:
            print(f"[WARN] Impossibile registrare l'ordine su DB: {e}")
    else:
        print("[WARN] Ordine non registrato nel DB per risposta di errore dall'exchange.")
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"

    color = GREEN if decision == "BUY" else RED if decision == "SELL" else YELLOW

    print(f"\n >>> {color}[EXECUTION - {source}] {decision} {action['pair']}{RESET}")

    if decision != "HOLD":
        final_qty = action.get('final_qty') if isinstance(action, dict) else getattr(action, "final_qty", 0.0)
        limit_price = action.get('limit_price') if isinstance(action, dict) else getattr(action, "limit_price", 0.0)
        take_profit = action.get('take_profit') if isinstance(action, dict) else getattr(action, "take_profit", 0.0)
        stop_loss = action.get('stop_loss') if isinstance(action, dict) else getattr(action, "stop_loss", 0.0)
        leverage = action.get('leverage') if isinstance(action, dict) else getattr(action, "leverage", 0.0)
        print(f"     Qty: {final_qty:.6f} @ {limit_price:.5f}")
        print(f"     TP:  {take_profit:.5f}")
        print(f"     SL:  {stop_loss:.5f}")
        print(f"     Lev: {leverage:.1f}x")
    print(" <<<\n")

def record_conflict_event(pair, high_side, low_side, counter):
    print(f"   [CONFLICT] {pair}: High={high_side} vs Low={low_side}. Count={counter}")

# ==============================================================================
# JOBS
# ==============================================================================

def run_high_tf_job(brain: BrainInstance, state_mgr: StateManager, all_pairs):
    print(f"\n=== AVVIO JOB HIGH-TF ({len(all_pairs)} pairs) ===")
    db = DatabaseManager()

    for pair in all_pairs:
        pair_name = pair['pair']
        base_currency = pair['base']

        try:
            context = db.get_trading_context(base_currency, TF_CONFIG_HIGH, with_orders=True)
        except Exception as e:
            continue

        _ensure_global_wallet_balance(context)
        if not context['candles'].get('1d'):
            continue

        action = brain.think(pair, context)

        if action:
            decision = action['decision']
            if decision in ["BUY", "SELL"]:
                print(f"[{datetime.now().strftime('%H:%M')}] HighTF {pair_name}: {decision}")
                state_mgr.update_watchlist(pair_name, decision)
            else:
                print(f"[{datetime.now().strftime('%H:%M')}] HighTF {pair_name}: HOLD")

    db.close_connection()
    print("=== FINE JOB HIGH-TF ===\n")


def run_low_tf_job(brain: BrainInstance, state_mgr: StateManager, all_pairs):
    print(f"\n=== AVVIO JOB LOW-TF ===")
    db = DatabaseManager()

    open_orders_rows = db.select_all("orders", "status = 'OPEN'")
    open_positions_pairs = set(row['pair'] for row in open_orders_rows)

    state_mgr.clean_watchlist(open_positions_pairs)
    target_pairs_names = set(state_mgr.watchlist.keys()).union(open_positions_pairs)
    target_pairs_dicts = [p for p in all_pairs if p['pair'] in target_pairs_names]

    if not target_pairs_dicts:
        print("   Nessuna coppia attiva (Watchlist vuota e nessun ordine aperto).")
        db.close_connection()
        return

    print(f"   Target Pairs: {len(target_pairs_dicts)} {list(target_pairs_names)}")

    for pair in target_pairs_dicts:
        pair_name = pair['pair']
        base_currency = pair['base']

        try:
            context = db.get_trading_context(base_currency, TF_CONFIG_LOW, with_orders=True)
        except Exception:
            continue

        _ensure_global_wallet_balance(context)
        if not context['candles'].get('1h'): continue

        action = brain.think(pair, context)
        if not action: continue

        decision_low = action['decision']
        print(f"[{datetime.now().strftime('%H:%M')}] LowTF {pair_name}: {decision_low}")

        open_order = context.get("order")
        if open_order and (open_order.get("status", "").upper() == "OPEN"):
            order_type_open = (open_order.get("orderType") or "").upper()
            if decision_low != "HOLD":
                if order_type_open == "LIMIT":
                    cnt = _increment_blocked_attempt(pair_name)
                    print(f"   [SKIP] {pair_name} ha gia un ordine LIMIT aperto. Tentativo {cnt}/3.")
                    if cnt >= 3:
                        runner_cancel = KrakenOrderRunner()
                        cancel_ok = runner_cancel.cancel_order(open_order)
                        if cancel_ok:
                            try:
                                db.updateOrder(open_order.get("id"), status="CLOSED", orderType="LIMIT")
                            except Exception as e:
                                print(f"[WARN] Aggiornamento DB dopo cancel fallito: {e}")
                            restore_cost = _safe_float(open_order.get("value_eur"), default=0.0)
                            if restore_cost == 0.0:
                                restore_cost = _safe_float(open_order.get("price_entry"), default=0.0) * _safe_float(open_order.get("qty"), default=0.0)
                            _update_wallet_on_cancel(restore_cost)
                            _reset_blocked_attempt(pair_name)
                        else:
                            print(f"   [WARN] Cancellazione Kraken non riuscita per {pair_name}.")
                else:
                    print(f"   [SKIP] {pair_name} ha gia un ordine aperto in esecuzione.")
                    _reset_blocked_attempt(pair_name)
            else:
                _reset_blocked_attempt(pair_name)
            continue
        else:
            _reset_blocked_attempt(pair_name)

        if decision_low == "HOLD":
            continue

        high_info = state_mgr.watchlist.get(pair_name)

        if not high_info:
            print(f"   [WARN] {pair_name} non ha decisione HighTF recente. Fallback LowTF.")
            execute_order(action, context=context, pair_info=pair, source="LowTF-Fallback")
            continue

        decision_high = high_info['last_high_decision']

        if decision_low == decision_high:
            print(f"   [MATCH] Coerenza confermata ({decision_low}).")
            execute_order(action, context=context, pair_info=pair, source="LowTF-Confirmed")
            state_mgr.reset_conflict(pair_name)
        else:
            cnt = state_mgr.increment_conflict(pair_name)
            record_conflict_event(pair_name, decision_high, decision_low, cnt)

            if cnt > 3:
                print(f"   >>> FORCE: Conflitto ripetuto {cnt} volte su {pair_name}. Vince LowTF.")
                execute_order(action, context=context, pair_info=pair, source="LowTF-Forced")
                state_mgr.reset_conflict(pair_name)
            else:
                print(f"   >>> BLOCCO: {pair_name} opposto a HighTF ({decision_high}). Attesa.")

    db.close_connection()
    print("=== FINE JOB LOW-TF ===\n")

# ==============================================================================
# MAIN
# ==============================================================================

def main_loop_dual_brain():
    print("==================================================")
    print("   AVVIO DUAL BRAIN AGENT (HighTF + LowTF)        ")
    print("==================================================")

    market_prov = MarketDataProvider()
    state_mgr = StateManager(STATE_FILE)

    brain_high = BrainInstance("HighTF", TF_CONFIG_HIGH, MODEL_PATH_HIGH)
    brain_low  = BrainInstance("LowTF", TF_CONFIG_LOW, MODEL_PATH_LOW)

    print("Caricamento coppie...")
    all_pairs = market_prov.getAllPairs(quote_filter="EUR", leverage_only=True)
    print(f"Trovate {len(all_pairs)} coppie EUR con leva.")

    last_high_run = None
    last_low_run = None

    while True:
        now = datetime.now()

        # Job HighTF (Ogni 60 minuti)
        if last_high_run is None or (now - last_high_run) >= timedelta(hours=1):
            try:
                last_high_run = datetime.now()
                run_high_tf_job(brain_high, state_mgr, all_pairs)
            except Exception as e:
                print(f"[CRITICAL ERROR] HighTF Crash: {e}")
                traceback.print_exc()

        # Job LowTF (Ogni 5 minuti)
        if last_low_run is None or (now - last_low_run) >= timedelta(minutes=5):
            try:
                last_low_run = datetime.now()
                run_low_tf_job(brain_low, state_mgr, all_pairs)
            except Exception as e:
                print(f"[CRITICAL ERROR] LowTF Crash: {e}")
                traceback.print_exc()

        time.sleep(10)

if __name__ == "__main__":
    try:
        main_loop_dual_brain()
    except KeyboardInterrupt:
        print("\n[STOP] Arresto manuale.")
