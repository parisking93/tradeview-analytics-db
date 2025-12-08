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
MODEL_PATH_HIGH = "trm_model_best.pth"
MODEL_PATH_LOW  = "trm_model_best_new_last.pth"

# Configurazione Timeframe
TF_CONFIG_HIGH = {"1d": 30, "4h": 50, "1h": 100}
TF_CONFIG_LOW  = {"1h": 30, "15m": 50, "5m": 100}

# Parametri del Thinking Loop
THINKING_STEPS = 6
MIN_STEPS = 2
HALT_THRESHOLD = 0.70

STATE_FILE = "dual_brain_state.json"

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
                hidden_dim=256
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
                wallet_balance=context_data['wallet_balance']
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

def execute_order(action, source="LowTF"):
    decision = action['decision']


    runner = KrakenOrderRunner()
    bodies = runner.build_bodies([action], validate=True, auto_brackets=False)
    test = runner.execute_bodies(bodies, timeout=0.8)
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"

    color = GREEN if decision == "BUY" else RED if decision == "SELL" else YELLOW

    print(f"\n >>> {color}[EXECUTION - {source}] {decision} {action['pair']}{RESET}")

    if decision != "HOLD":
        print(f"     Qty: {action['final_qty']:.6f} @ {action['limit_price']:.5f}")
        print(f"     TP:  {action['take_profit']:.5f}")
        print(f"     SL:  {action['stop_loss']:.5f}")
        print(f"     Lev: {action['leverage']:.1f}x")
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
            context = db.get_trading_context(base_currency, TF_CONFIG_HIGH)
        except Exception as e:
            continue

        if not context['candles'].get('1d'):
            continue

        action = brain.think(pair, context)

        if action:
            decision = action['decision']
            if decision in ["BUY", "SELL"]:
                print(f"[{datetime.now().strftime('%H:%M')}] HighTF {pair_name}: {decision}")
                state_mgr.update_watchlist(pair_name, decision)

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

        if not context['candles'].get('1h'): continue

        action = brain.think(pair, context)
        if not action: continue

        decision_low = action['decision']
        print(f"[{datetime.now().strftime('%H:%M')}] LowTF {pair_name}: {decision_low}")

        if decision_low == "HOLD":
            continue

        high_info = state_mgr.watchlist.get(pair_name)

        if not high_info:
            print(f"   [WARN] {pair_name} non ha decisione HighTF recente. Fallback LowTF.")
            execute_order(action, source="LowTF-Fallback")
            continue

        decision_high = high_info['last_high_decision']

        if decision_low == decision_high:
            print(f"   [MATCH] Coerenza confermata ({decision_low}).")
            execute_order(action, source="LowTF-Confirmed")
            state_mgr.reset_conflict(pair_name)
        else:
            cnt = state_mgr.increment_conflict(pair_name)
            record_conflict_event(pair_name, decision_high, decision_low, cnt)

            if cnt > 3:
                print(f"   >>> FORCE: Conflitto ripetuto {cnt} volte su {pair_name}. Vince LowTF.")
                execute_order(action, source="LowTF-Forced")
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
                run_high_tf_job(brain_high, state_mgr, all_pairs)
                last_high_run = datetime.now()
            except Exception as e:
                print(f"[CRITICAL ERROR] HighTF Crash: {e}")
                traceback.print_exc()

        # Job LowTF (Ogni 5 minuti)
        if last_low_run is None or (now - last_low_run) >= timedelta(minutes=5):
            try:
                run_low_tf_job(brain_low, state_mgr, all_pairs)
                last_low_run = datetime.now()
            except Exception as e:
                print(f"[CRITICAL ERROR] LowTF Crash: {e}")
                traceback.print_exc()

        time.sleep(10)

if __name__ == "__main__":
    try:
        main_loop_dual_brain()
    except KeyboardInterrupt:
        print("\n[STOP] Arresto manuale.")
