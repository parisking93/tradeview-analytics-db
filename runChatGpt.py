import os
import sys
import time
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple

import torch

# Ensure local modules are importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from db.DatabaseManager import DatabaseManager
from db.MarketDataProvider import MarketDataProvider
from trading.Vectorizer import DataVectorizer, VectorizerConfig
from trading.TrmAgent import MultiTimeframeTRM
from trading.Decoder import ActionDecoder

# =============================================================================
# CONFIGURAZIONE DI BASE
# =============================================================================

QUOTE_FILTER = "EUR"
LEVERAGE_ONLY = True

# Timeframe config per i due cervelli
TF_CONFIG_HIGH = {
    "1d": 60,   # TODO: calibra in base al tuo Vectorizer / storico reale
    "4h": 120,
    "1h": 120,
}

TF_CONFIG_LOW = {
    "1h": 30,
    "15m": 50,
    "5m": 100,
}

# File dei pesi (uno per cervello)
MODEL_PATH_HIGH = "trm_model_best.pth"
MODEL_PATH_LOW = "trm_model_best_long.pth"

# Parametri del thinking loop (ACT)
THINKING_STEPS_HIGH = 6
THINKING_STEPS_LOW = 6
MIN_STEPS_HIGH = 2
MIN_STEPS_LOW = 2
HALT_THRESHOLD_HIGH = 0.75
HALT_THRESHOLD_LOW = 0.75

# Watchlist / conflitti
WATCHLIST_MAX_AGE_MINUTES = 60
MAX_CONFLICT_BEFORE_FORCE = 3

STATE_FILE = "dual_brain_state.json"


# =============================================================================
# UTILITY LOGGING
# =============================================================================

def log(brain: str, message: str) -> None:
    """
    Log con timestamp e nome cervello (o modulo).
    """
    ts = datetime.now().isoformat(timespec="seconds")
    print(f"[{ts}] [{brain}] {message}")


# =============================================================================
# PERSISTENZA MINIMA (watchlist + conflict_counters)
# =============================================================================

def _serialize_watchlist(watchlist: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    serialized: Dict[str, Dict[str, Any]] = {}
    for pair, data in watchlist.items():
        entry = dict(data)
        added_at = entry.get("added_at")
        if isinstance(added_at, datetime):
            entry["added_at"] = added_at.isoformat()
        serialized[pair] = entry
    return serialized


def _deserialize_watchlist(raw: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    watchlist: Dict[str, Dict[str, Any]] = {}
    for pair, data in raw.items():
        entry = dict(data)
        added_at = entry.get("added_at")
        if isinstance(added_at, str):
            try:
                entry["added_at"] = datetime.fromisoformat(added_at)
            except Exception:
                entry["added_at"] = datetime.now()
        watchlist[pair] = entry
    return watchlist


def load_state_from_disk() -> Tuple[Dict[str, Dict[str, Any]], Dict[str, int]]:
    """
    Carica watchlist_small_tf e conflict_counters da disco.
    Se il file non esiste o è invalido, restituisce strutture vuote.
    """
    if not os.path.exists(STATE_FILE):
        return {}, {}

    try:
        with open(STATE_FILE, "r") as f:
            raw = json.load(f)
        watchlist = _deserialize_watchlist(raw.get("watchlist_small_tf", {}))
        conflict_counters = raw.get("conflict_counters", {})
        # normalizza tipi numerici
        conflict_counters = {k: int(v) for k, v in conflict_counters.items()}
        log("STATE", f"Stato caricato da {STATE_FILE}. "
                     f"Watchlist: {len(watchlist)} pair, Conflitti: {len(conflict_counters)}")
        return watchlist, conflict_counters
    except Exception as e:
        log("STATE", f"Errore nel loading dello stato: {e}. Uso stato vuoto.")
        return {}, {}


def save_state_to_disk(watchlist_small_tf: Dict[str, Dict[str, Any]],
                       conflict_counters: Dict[str, int]) -> None:
    """
    Salva su disco lo stato minimo (watchlist + counters).
    """
    try:
        state = {
            "watchlist_small_tf": _serialize_watchlist(watchlist_small_tf),
            "conflict_counters": conflict_counters,
        }
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
        log("STATE", f"Stato salvato su {STATE_FILE}.")
    except Exception as e:
        log("STATE", f"Errore nel salvataggio dello stato: {e}")
        # TODO: se serve robustezza, qui si può aggiungere un meccanismo di backup/rotazione.


# =============================================================================
# ACCESSO AL DB / CONTESTO
# =============================================================================

def load_trading_context(pair: Dict[str, Any], tf_config: Dict[str, int]) -> Optional[Dict[str, Any]]:
    """
    Carica il contesto di trading dal DB per una certa pair e configurazione timeframe.
    Wrappa la logica già usata in RunRefinedAgent.run_brain_cycle.
    """
    currency = pair.get("base")
    pair_name = pair.get("pair")
    try:
        db = DatabaseManager()
        context = db.get_trading_context(currency, tf_config)
        db.close_connection()
    except Exception as e:
        log("CTX", f"[{pair_name}] Errore DB: {e}")
        return None

    # Controllo minimo: se manca un timeframe chiave, skippa
    # NB: qui assumiamo che 1h sia sempre presente; se per l'high TF preferisci 4h/1d, puoi estendere il check.
    if not context["candles"].get("1h"):
        log("CTX", f"[{pair_name}] Dati insufficienti per 1h, skip.")
        return None

    return context


def has_open_position(context: Dict[str, Any]) -> bool:
    """
    Determina se esiste una posizione aperta sulla pair usando il contesto DB.
    Per ora è un'implementazione molto semplice:

      - Se context['order'] è non-null / non-vuoto, assumiamo ci sia una posizione aperta.

    TODO: raffinamento in base alla struttura reale dell'ordine (qty > 0, stato, ecc.).
    """
    order = context.get("order")
    if order is None:
        return False
    # Se è un dict/list non vuoto, lo consideriamo una posizione aperta.
    if isinstance(order, dict) and order:
        return True
    if isinstance(order, list) and len(order) > 0:
        return True
    return False


# =============================================================================
# CERVELLO GENERICO SU TIMEFRAME (HighTF_Brain / LowTF_Brain)
# =============================================================================

class TimeframeBrain:
    """
    Incapsula:
      - tf_config
      - modello MultiTimeframeTRM + pesi
      - parametri del thinking loop
      - vectorizer config

    Espone run_for_pair(pair) che ritorna:
      final_action (dict), heads (dict), altre info di debug.
    """

    def __init__(
        self,
        name: str,
        tf_config: Dict[str, int],
        model_path: str,
        device: Optional[torch.device] = None,
        thinking_steps: int = 6,
        min_steps: int = 2,
        halt_threshold: float = 0.75,
    ):
        self.name = name
        self.tf_config = tf_config
        self.model_path = model_path
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.thinking_steps = thinking_steps
        self.min_steps = min_steps
        self.halt_threshold = halt_threshold

        self.vectorizer_config = VectorizerConfig(candle_history_config=self.tf_config)

        self.model: Optional[MultiTimeframeTRM] = None
        self._model_input_dim: Optional[int] = None
        self._model_static_dim: Optional[int] = None

    # -------------------------------------------------------------------------
    # Inizializzazione lazy del modello (una sola volta per cervello)
    # -------------------------------------------------------------------------
    def _ensure_model(self, input_dim_candle: int, static_dim: int) -> None:
        if self.model is not None:
            # Facoltativo: assert sulle dimensioni per sicurezza
            if self._model_input_dim != input_dim_candle or self._model_static_dim != static_dim:
                log(self.name, "WARNING: dimensioni input/static cambiate rispetto al modello caricato.")
            return

        log(self.name, f"Inizializzo modello MultiTimeframeTRM "
                       f"(input_dim_candle={input_dim_candle}, static_dim={static_dim})")
        model = MultiTimeframeTRM(
            tf_configs=self.tf_config,
            input_size_per_candle=input_dim_candle,
            static_size=static_dim,
            hidden_dim=256,
        ).to(self.device)

        if not os.path.exists(self.model_path):
            log(self.name, f"ATTENZIONE: file pesi {self.model_path} non trovato. "
                           f"Uso pesi random (decisioni non affidabili).")
        else:
            state = torch.load(self.model_path, map_location=self.device)
            model.load_state_dict(state)
            log(self.name, f"Pesi caricati da {self.model_path}.")

        model.eval()
        self.model = model
        self._model_input_dim = input_dim_candle
        self._model_static_dim = static_dim

    # -------------------------------------------------------------------------
    # Thinking loop (Refinement / ACT) per una singola pair
    # -------------------------------------------------------------------------
    def run_for_pair(self, pair: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]],
                                                          Optional[Dict[str, Any]],
                                                          Dict[str, Any]]:
        """
        Esegue l'intero ciclo:
          - carica contesto dal DB
          - vettorializza
          - esegue il refinement loop sul MultiTimeframeTRM
          - decodifica l'azione finale con ActionDecoder

        Ritorna:
          final_action (dict o None),
          last_heads (dict o None),
          debug_info (dict)
        """
        pair_name = pair.get("pair")
        currency = pair.get("base")

        log(self.name, f"Avvio ciclo cervello per {pair_name} (base={currency})")

        context = load_trading_context(pair, self.tf_config)
        if context is None:
            return None, None, {"reason": "no_context"}

        # Vettorializzazione
        log(self.name, f"{pair_name}: vettorializzazione in corso...")
        vectorizer = DataVectorizer(self.vectorizer_config)
        try:
            inputs, ref_price = vectorizer.vectorize(
                candles_db_data=context["candles"],
                open_order=context["order"],
                forecast_db_data=context["forecast"],
                pair_limits=pair.get("pair_limits"),
                wallet_balance=context["wallet_balance"],
            )
        except Exception as e:
            log(self.name, f"{pair_name}: errore in vectorize(): {e}")
            return None, None, {"reason": "vectorize_error", "error": str(e)}

        static_dim = vectorizer.static_total_dim
        input_dim_candle = vectorizer.candle_dim

        self._ensure_model(input_dim_candle=input_dim_candle, static_dim=static_dim)
        if self.model is None:
            log(self.name, f"{pair_name}: modello non disponibile, skip.")
            return None, None, {"reason": "no_model"}

        decoder = ActionDecoder(ref_price, pair)

        h = None
        steps_taken = 0
        final_action: Optional[Dict[str, Any]] = None
        last_heads: Optional[Dict[str, Any]] = None

        with torch.no_grad():
            for k in range(self.thinking_steps):
                steps_taken = k + 1
                y, h = self.model(inputs, h)
                heads = self.model.get_heads_dict(y)
                last_heads = heads

                halt_prob = heads["halt_prob"].item()
                log(self.name, f"{pair_name}: Step {steps_taken}/{self.thinking_steps} "
                               f"halt_prob={halt_prob:.1%}")

                temp_action = decoder.decode(heads, steps_taken)
                decoder.print_action(temp_action, is_final=False)

                can_stop = steps_taken >= self.min_steps
                wants_to_stop = halt_prob >= self.halt_threshold
                forced_stop = steps_taken == self.thinking_steps

                if can_stop and (wants_to_stop or forced_stop):
                    if wants_to_stop:
                        log(self.name, f"{pair_name}: HALTING (halt_prob={halt_prob:.1%})")
                    else:
                        log(self.name, f"{pair_name}: MAX STEPS raggiunto, stop forzato.")
                    final_action = temp_action
                    break

            if final_action is None and last_heads is not None:
                final_action = decoder.decode(last_heads, self.thinking_steps)

        if final_action is not None:
            decoder.print_action(final_action, is_final=True)
        else:
            log(self.name, f"{pair_name}: nessuna azione finale generata.")

        debug_info = {
            "steps_taken": steps_taken,
            "currency": currency,
            "pair": pair_name,
        }

        return final_action, last_heads, debug_info


# =============================================================================
# FUNZIONE GENERICA run_trm_for_pair (riusabile anche fuori da questo file)
# =============================================================================

def run_trm_for_pair(
    pair: Dict[str, Any],
    tf_config: Dict[str, int],
    model_path: str,
    device: Optional[torch.device],
    context_loader,
    vectorizer_config: VectorizerConfig,
    decoder_class,
    thinking_steps: int,
    min_steps: int,
    halt_threshold: float,
):
    """
    Wrapper generico per eseguire un ciclo TRM su una pair.
    Internamente usa TimeframeBrain per sfruttare la stessa logica del refinement loop.

    NOTA:
    - context_loader è una funzione tipo load_trading_context(pair, tf_config)
    - decoder_class è tipicamente ActionDecoder
    """
    # Per semplicità creiamo un cervello ad-hoc che usa il context_loader di questo modulo.
    brain = TimeframeBrain(
        name="GenericTRM",
        tf_config=tf_config,
        model_path=model_path,
        device=device,
        thinking_steps=thinking_steps,
        min_steps=min_steps,
        halt_threshold=halt_threshold,
    )

    # Override del loader di contesto via monkey-patching, se si vuole usare quello passato.
    # Qui teniamo l'implementazione semplice: ignoriamo context_loader e
    # usiamo load_trading_context direttamente. Se vuoi sfruttare context_loader,
    # puoi adattare TimeframeBrain.run_for_pair ad accettare un loader esterno.
    # TODO: integrare context_loader in TimeframeBrain se necessario.

    final_action, heads, debug_info = brain.run_for_pair(pair)
    return final_action, heads, debug_info


# =============================================================================
# GESTIONE WATCHLIST (timeframe piccoli)
# =============================================================================

def update_watchlist_from_high_tf(
    watchlist_small_tf: Dict[str, Dict[str, Any]],
    pair_name: str,
    high_decision: str,
    now: datetime,
) -> None:
    """
    Aggiorna/crea l'entry di watchlist per una pair data una decisione BUY/SELL
    dell'HighTF_Brain.
    """
    entry = watchlist_small_tf.get(pair_name, {})
    entry["added_at"] = now
    entry["last_high_decision"] = high_decision
    # Se non sappiamo ancora se c'è una posizione aperta, la lasciamo invariata o False.
    entry.setdefault("has_open_position", False)
    watchlist_small_tf[pair_name] = entry
    log("WATCHLIST", f"{pair_name}: aggiornato da HighTF (decision={high_decision}).")


def prune_watchlist(
    watchlist_small_tf: Dict[str, Dict[str, Any]],
    now: datetime,
    max_age_minutes: int = WATCHLIST_MAX_AGE_MINUTES,
) -> None:
    """
    Rimuove dalla watchlist le pair scadute (added_at > max_age) e senza posizione aperta.
    """
    to_delete = []
    for pair_name, entry in watchlist_small_tf.items():
        added_at = entry.get("added_at")
        has_pos = entry.get("has_open_position", False)

        if isinstance(added_at, datetime):
            age = now - added_at
        else:
            # In caso di dato corrotto, rimuoviamo con prudenza.
            age = timedelta(days=999)

        if age > timedelta(minutes=max_age_minutes) and not has_pos:
            to_delete.append(pair_name)

    for pair_name in to_delete:
        watchlist_small_tf.pop(pair_name, None)
        log("WATCHLIST", f"{pair_name}: rimosso per scadenza e nessuna posizione aperta.")


# =============================================================================
# GESTIONE CONFLITTI HIGH vs LOW
# =============================================================================

def record_conflict(pair_name: str, high_side: str, low_side: str, forced: bool = False) -> None:
    """
    Placeholder per registrare un conflitto fra cervelli.
    Per ora fa solo logging a console, ma può essere esteso per scrivere su DB/file.
    """
    flag = "FORCED" if forced else "PENDING"
    log("CONFLICT", f"{pair_name}: conflitto HighTF={high_side} vs LowTF={low_side} [{flag}]")
    # TODO: persistere su DB/file per analisi futura.


def handle_conflict_for_pair(
    pair_name: str,
    high_side: Optional[str],
    low_side: str,
    conflict_counters: Dict[str, int],
    max_conflicts: int = MAX_CONFLICT_BEFORE_FORCE,
) -> Tuple[bool, bool]:
    """
    Applica le regole di coerenza tra cervelli per una singola pair.

    Ritorna:
      (should_execute, forced_by_conflict)
    """
    # Caso C: HighTF non ha decisione valida → trattiamo come pair libera
    # (eseguiamo sempre la decisione LowTF ma logghiamo un warning).
    if not high_side:
        log("CONFLICT", f"{pair_name}: nessuna last_high_decision, eseguo LowTF come libero.")
        return True, False

    # Caso A: stessa direzione → esegui normalmente e resetta contatore
    if high_side == low_side:
        conflict_counters[pair_name] = 0
        log("CONFLICT", f"{pair_name}: HighTF e LowTF coerenti ({high_side}). Eseguo.")
        return True, False

    # Caso B: direzioni opposte → conflitto
    current = conflict_counters.get(pair_name, 0) + 1
    conflict_counters[pair_name] = current
    record_conflict(pair_name, high_side, low_side, forced=False)

    if current > max_conflicts:
        # Forziamo l'esecuzione LowTF e resettiamo il contatore
        conflict_counters[pair_name] = 0
        record_conflict(pair_name, high_side, low_side, forced=True)
        log("CONFLICT", f"{pair_name}: conflitto ripetuto {current} volte, "
                        f"forzo esecuzione LowTF ({low_side}).")
        return True, True

    log("CONFLICT", f"{pair_name}: conflitto #{current} (HighTF={high_side}, LowTF={low_side}), "
                    f"non eseguo ancora.")
    return False, False


# =============================================================================
# ESECUZIONE AZIONI (placeholder executor)
# =============================================================================

def execute_low_tf_action(action: Dict[str, Any], pair_name: str, forced_by_conflict: bool = False) -> None:
    """
    Qui colleghi il decoder/azione al tuo executor reale (API exchange).
    Per ora è solo un placeholder con logging.
    """
    decision = action.get("decision")
    msg = f"{pair_name}: eseguo azione LowTF={decision}"
    if forced_by_conflict:
        msg += " [FORCED_BY_CONFLICT]"
    log("EXEC", msg)
    # TODO: integra il tuo executor reale (es. KrakenOrderRunner, ecc.)


# =============================================================================
# JOB HighTF_Brain
# =============================================================================

def run_high_tf_cycle(
    brain: TimeframeBrain,
    pairs: Any,
    watchlist_small_tf: Dict[str, Dict[str, Any]],
) -> None:
    """
    Job HighTF_Brain:
      - gira ogni ora
      - per ogni pair:
          * esegue TimeframeBrain.run_for_pair
          * se decisione BUY/SELL → aggiorna watchlist_small_tf
    """
    now = datetime.now()
    log(brain.name, "Avvio ciclo HighTF_Brain...")
    for pair in pairs:
        pair_name = pair.get("pair")

        final_action, heads, debug_info = brain.run_for_pair(pair)
        if final_action is None:
            continue

        decision = final_action.get("decision")
        if decision in ("BUY", "SELL"):
            update_watchlist_from_high_tf(
                watchlist_small_tf=watchlist_small_tf,
                pair_name=pair_name,
                high_decision=decision,
                now=now,
            )
        else:
            # HOLD: non rimuoviamo dalla watchlist; sarà prune_watchlist a farlo.
            log(brain.name, f"{pair_name}: decisione HIGH={decision}, nessun update watchlist.")

    log(brain.name, "Fine ciclo HighTF_Brain.")


# =============================================================================
# JOB LowTF_Brain
# =============================================================================

def run_low_tf_cycle(
    brain: TimeframeBrain,
    pairs: Any,
    watchlist_small_tf: Dict[str, Dict[str, Any]],
    conflict_counters: Dict[str, int],
) -> None:
    """
    Job LowTF_Brain:
      - gira ogni 5 minuti
      - lavora su:
          * tutte le pair in watchlist_small_tf non scadute
          * tutte le pair con posizioni aperte
      - applica la logica di coerenza con HighTF_Brain per le azioni BUY/SELL.
    """
    now = datetime.now()
    log(brain.name, "Avvio ciclo LowTF_Brain...")

    # 1) Pulisce la watchlist da voci scadute e senza posizione
    prune_watchlist(watchlist_small_tf, now)

    # 2) Costruisce l'elenco di pair su cui lavorare
    pairs_by_name = {p["pair"]: p for p in pairs}

    target_pairs: Dict[str, Dict[str, Any]] = {}

    # a) Pair in watchlist non scadute (dopo prune)
    for pair_name in watchlist_small_tf.keys():
        if pair_name in pairs_by_name:
            target_pairs[pair_name] = pairs_by_name[pair_name]

    # b) Pair con posizioni aperte (anche se non in watchlist)
    for pair in pairs:
        pair_name = pair["pair"]
        context = load_trading_context(pair, brain.tf_config)
        if context is None:
            continue

        pos_open = has_open_position(context)
        if pos_open:
            target_pairs[pair_name] = pair

        # Aggiorna flag has_open_position in watchlist se presente
        if pair_name in watchlist_small_tf:
            entry = watchlist_small_tf[pair_name]
            entry["has_open_position"] = bool(pos_open)
            watchlist_small_tf[pair_name] = entry

    # 3) Esegue il cervello LowTF su ogni pair target
    for pair_name, pair in target_pairs.items():
        final_action, heads, debug_info = brain.run_for_pair(pair)
        if final_action is None:
            continue

        low_decision = final_action.get("decision")
        if low_decision == "HOLD":
            log(brain.name, f"{pair_name}: LowTF=HOLD, nessuna azione.")
            continue

        if low_decision not in ("BUY", "SELL"):
            log(brain.name, f"{pair_name}: LowTF decisione non riconosciuta ({low_decision}), skip.")
            continue

        high_decision = None
        if pair_name in watchlist_small_tf:
            high_decision = watchlist_small_tf[pair_name].get("last_high_decision")

        should_execute, forced = handle_conflict_for_pair(
            pair_name=pair_name,
            high_side=high_decision,
            low_side=low_decision,
            conflict_counters=conflict_counters,
        )

        if should_execute:
            execute_low_tf_action(final_action, pair_name, forced_by_conflict=forced)

    log(brain.name, "Fine ciclo LowTF_Brain.")


# =============================================================================
# LOOP PRINCIPALE (stile mainLoop.py)
# =============================================================================

def main_loop_dual_brain() -> None:
    """
    Loop infinito che orchestra:
      - HighTF_Brain (1d,4h,1h) ogni ora
      - LowTF_Brain  (1h,15m,5m) ogni 5 minuti

    Riutilizza:
      - DatabaseManager
      - MarketDataProvider
      - DataVectorizer / VectorizerConfig
      - MultiTimeframeTRM
      - ActionDecoder

    E mantiene in memoria:
      - watchlist_small_tf
      - conflict_counters

    Con persistenza minima su disco (STATE_FILE).
    """
    log("MAIN", "Avvio main_loop_dual_brain (dual MultiTimeframeTRM)...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Carica la lista delle coppie una sola volta all'avvio (puoi aggiornare periodicamente se preferisci)
    market_prov = MarketDataProvider()
    pairs = market_prov.getAllPairs(quote_filter=QUOTE_FILTER, leverage_only=LEVERAGE_ONLY)
    log("MAIN", f"Caricate {len(pairs)} coppie dal MarketDataProvider.")

    # Inizializza i due cervelli
    high_brain = TimeframeBrain(
        name="HighTF_Brain",
        tf_config=TF_CONFIG_HIGH,
        model_path=MODEL_PATH_HIGH,
        device=device,
        thinking_steps=THINKING_STEPS_HIGH,
        min_steps=MIN_STEPS_HIGH,
        halt_threshold=HALT_THRESHOLD_HIGH,
    )

    low_brain = TimeframeBrain(
        name="LowTF_Brain",
        tf_config=TF_CONFIG_LOW,
        model_path=MODEL_PATH_LOW,
        device=device,
        thinking_steps=THINKING_STEPS_LOW,
        min_steps=MIN_STEPS_LOW,
        halt_threshold=HALT_THRESHOLD_LOW,
    )

    # Stato in memoria
    watchlist_small_tf, conflict_counters = load_state_from_disk()
    last_high_tf_run: Optional[datetime] = None
    last_low_tf_run: Optional[datetime] = None

    while True:
        now = datetime.now()

        # Job HighTF_Brain ogni ora
        if last_high_tf_run is None or (now - last_high_tf_run) >= timedelta(hours=1):
            run_high_tf_cycle(high_brain, pairs, watchlist_small_tf)
            last_high_tf_run = now
            save_state_to_disk(watchlist_small_tf, conflict_counters)

        # Job LowTF_Brain ogni 5 minuti
        if last_low_tf_run is None or (now - last_low_tf_run) >= timedelta(minutes=5):
            run_low_tf_cycle(low_brain, pairs, watchlist_small_tf, conflict_counters)
            last_low_tf_run = now
            save_state_to_disk(watchlist_small_tf, conflict_counters)

        # Piccola pausa per non saturare la CPU
        time.sleep(5)


if __name__ == "__main__":
    main_loop_dual_brain()
