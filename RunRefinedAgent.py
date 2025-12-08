import torch
import torch.nn.functional as F
import sys
import os
import json # Serve per parsare l'array dal DB

# Assicurati che i moduli siano nel path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from db.DatabaseManager import DatabaseManager
from trading.Vectorizer import DataVectorizer, VectorizerConfig
from trading.TrmAgent import MultiTimeframeTRM
from db.MarketDataProvider import MarketDataProvider
from trading.Decoder import ActionDecoder
# ==============================================================================
# 2. PROCESSO DI TRADING "METICOLOSO"
# ==============================================================================
def run_brain_cycle(pair: dict, tf_config: dict = {"1h":30, "15m":50, "5m":100}):
    # --- CONFIGURAZIONE ---
    CURRENCY = pair.get('base')
    PAIR = pair.get('pair')

    # Quanti passi di "riflessione" deve fare il modello prima di decidere?
    # Più alto è, più "stabile" è la decisione (simile al cervello umano che ci pensa su).
    THINKING_STEPS = 6
    MIN_STEPS = 2           # Minimo sindacale per stabilizzarsi
    HALT_THRESHOLD = 0.75      # Abbassato per evitare stop troppo precoci
    # Configurazione Timeframe (Deve combaciare con Vectorizer e Modello)
    tf_config = tf_config

    print(f"\n--- 1. AVVIO SISTEMA PER {PAIR} ---")

    # A. CONNESSIONE DB
    try:
        db = DatabaseManager()
        # Recuperiamo tutto il contesto in una sola chiamata ottimizzata
        context = db.get_trading_context(CURRENCY, tf_config)
        db.close_connection()
    except Exception as e:
        print(f"[ERROR] Database fallito: {e}")
        return

    # Controllo dati minimi
    if not context['candles'].get('1h'):
        print("[WARN] Dati insufficienti per elaborare una strategia.")
        return

    # B. VETTORIALIZZAZIONE (La "Vista")
    # Qui avviene la magia: tutte le colonne (stringhe, date, numeri) diventano tensori.
    print("--- 2. VETTORIALIZZAZIONE ---")
    vec_config = VectorizerConfig(candle_history_config=tf_config)
    vectorizer = DataVectorizer(vec_config)

    # inputs contiene: 'seq_1d', 'seq_1h', 'seq_forecast', 'static'
    # ref_price serve per denormalizzare i prezzi alla fine
    inputs, ref_price = vectorizer.vectorize(
        candles_db_data=context['candles'],
        open_order=context['order'],
        forecast_db_data=context['forecast'],
        pair_limits=pair.get('pair_limits'),
        wallet_balance=context['wallet_balance']
    )

    # Nota tecnica: Vectorizer.py calcola dinamicamente le dimensioni.
    # Dobbiamo passarle al modello affinché sappia quanto sono grandi i vettori in ingresso.
    static_dim = vectorizer.static_total_dim
    input_dim_candle = vectorizer.candle_dim

    print(f"   Input Dimension Candle: {input_dim_candle}")
    print(f"   Input Dimension Static: {static_dim}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # C. INIZIALIZZAZIONE MODELLO (Il "Cervello")
    print("--- 3. CARICAMENTO MODELLO ---")
    model = MultiTimeframeTRM(
        tf_configs=tf_config,
        input_size_per_candle=input_dim_candle,
        static_size=static_dim,
        hidden_dim=256 # Deve essere lo stesso usato in fase di training
    ).to(device)

    # !IMPORTANTE!: Qui dovresti caricare i pesi addestrati.
    model.load_state_dict(torch.load("trm_model_best_long.pth", map_location=device))
    # Per ora usiamo pesi casuali inizializzati, quindi le decisioni saranno randomiche.
    model.eval()

    # D. IL THINKING LOOP (Il Cuore della richiesta)
    print(f"--- 4. CICLO DI RIFLESSIONE (K={THINKING_STEPS}) ---")

    decoder = ActionDecoder(ref_price, pair)

    # Memoria a breve termine (Hidden State) inizializzata a None (o Zeri)
    h = None
    steps_taken = 0
    final_action = None

    with torch.no_grad():
        # Ciclo di raffinamento (Refinement Loop)
        for k in range(THINKING_STEPS):
            steps_taken = k + 1
            # Passiamo gli STESSI input visivi, ma la memoria 'h' evolve.
            # Il modello combina ciò che vede (inputs) con ciò che ha pensato prima (h).
            y, h = model(inputs, h)
            current_heads = model.get_heads_dict(y)
            # Estraiamo le "Heads" (le intenzioni correnti)
            halt_prob = current_heads['halt_prob'].item()
            print(f"[Step {steps_taken}] Halt Prob: {halt_prob:.1%}")

            can_stop = (steps_taken >= MIN_STEPS)
            wants_to_stop = (halt_prob >= HALT_THRESHOLD)
            forced_stop = (steps_taken == THINKING_STEPS)
            # (Opzionale) Possiamo vedere cosa pensa durante il processo
            temp_action = decoder.decode(current_heads, k + 1)
            decoder.print_action(temp_action, is_final=False)
            if can_stop and (wants_to_stop or forced_stop):
                if wants_to_stop:
                    print(f"   >>> HALTING: Il modello è convinto ({halt_prob:.1%}).")
                else:
                    print(f"   >>> MAX STEPS: Limite raggiunto.")

                final_action = temp_action
                break
            # print(temp_action)
            # Decommenta per vedere i pensieri intermedi:
            # decoder.print_action(temp_action, is_final=False)

        # E. DECISIONE FINALE
        # Dopo K passi, la memoria 'h' è stabile. L'ultimo output 'y' è la decisione ponderata.
        if final_action is None:
            final_heads = model.get_heads_dict(y)
            final_action = decoder.decode(final_heads, THINKING_STEPS)

        decoder.print_action(final_action, is_final=True)

    # F. POST-PROCESSING (Esecuzione Ordine)
    # Qui andrebbe il codice che prende 'final_action' e chiama l'API dell'Exchange
    if final_action['decision'] != "HOLD":
        print(f"\n[EXECUTION] Preparazione ordine {final_action['decision']} su {PAIR}...")
        # execute_order(final_action)
    else:
        print("\n[EXECUTION] Nessuna operazione richiesta. Attesa prossima candela.")

if __name__ == "__main__":
    market_prov = MarketDataProvider()

    all_pairs_eur = market_prov.getAllPairs(quote_filter="EUR", leverage_only=True)
    for p in all_pairs_eur:
        run_brain_cycle(p)
