import os
import krakenex
from dotenv import load_dotenv

# 1. Carica le variabili d'ambiente dal file .env
load_dotenv()
api_key = os.getenv('KRAKEN_KEY')
api_secret = os.getenv('KRAKEN_SECRET')

# 2. Inizializza l'API di Kraken
k = krakenex.API(key=api_key, secret=api_secret)


# --- MAPPE DI CONFIGURAZIONE ---
DISPLAY_MAP = {
    'XXBT': 'Bitcoin (BTC)',
    'XBT.B': 'Bitcoin (BTC)',
    'XETH': 'Ethereum (ETH)',
    'ETH2': 'Ethereum (ETH Staked)',
    'ETH2.S': 'Ethereum (Reward)',
    'XDOT': 'Polkadot (DOT)',
    'SOL': 'Solana (SOL)',
    'MATIC': 'Polygon (MATIC)',
    'ADA': 'Cardano (ADA)',
}

# Associa l'asset del saldo all'asset usato nei Trade (Es. XBT.B deriva da XXBT)
HISTORY_MAPPING = {
    'XBT.B': 'XXBT',
    'ETH2': 'XETH',
    'ETH2.S': 'XETH',
}

def get_trades_history_summary(current_eur_usd_rate=1.05):
    """
    Scarica TUTTO lo storico.
    Raggruppa per ASSET BASE (es. tutto ciò che è Bitcoin finisce insieme).
    Converte i costi USD in EUR per darti un prezzo medio omogeneo.
    """
    print("--- Download Storico Completo (Analisi incrociata valute)... ---")

    all_trades = {}
    offset = 0

    # --- 1. DOWNLOAD PAGINATO ---
    while True:
        try:
            # Scarica a blocchi
            response = k.query_private('TradesHistory', {'type': 'no position', 'limit': 1000, 'ofs': offset})
            if response.get('error'):
                print(f"Errore download storico: {response['error']}")
                break

            result = response.get('result', {})
            trades = result.get('trades', {})

            if not trades: break

            all_trades.update(trades)
            offset = result.get('last') # Aggiorna cursore per pagina successiva

            # Se scarica meno del limite, abbiamo finito
            if len(trades) < 1000: break

        except Exception as e:
            print(f"Errore loop: {e}")
            break

    print(f" -> Analisi di {len(all_trades)} trade totali.")

    # --- 2. AGGREGAZIONE INTELLIGENTE ---
    # Chiave: Asset Base (es. XXBT), Valore: {cost_eur: ..., vol: ...}
    summary = {}

    for txid, info in all_trades.items():
        if info['type'] == 'buy':
            pair = info['pair']     # Es. XXBTZUSD, XXBTZEUR
            cost = float(info['cost'])
            vol = float(info['vol'])

            # Identificazione Asset Base e Valuta usata
            base_asset = None
            is_usd_purchase = False

            # Euristiche per capire cosa stiamo comprando
            if 'XBT' in pair or 'BTC' in pair: base_asset = 'XXBT'
            elif 'ETH' in pair: base_asset = 'XETH'
            elif 'DOT' in pair: base_asset = 'XDOT'
            elif 'SOL' in pair: base_asset = 'SOL'
            elif 'ADA' in pair: base_asset = 'ADA'
            elif 'MATIC' in pair: base_asset = 'MATIC'
            else:
                # Fallback: Prende i primi 4 caratteri se inizia con X, o 3 caratteri
                if pair.startswith('X') and len(pair) >= 8: base_asset = pair[:4]
                else: base_asset = pair[:3] # Molto approssimativo

            # Controllo Valuta di pagamento
            if 'USD' in pair:
                is_usd_purchase = True

            # Normalizzazione Costo in EURO
            cost_in_eur = cost
            if is_usd_purchase:
                # Se ho speso 100 USD e il cambio è 1.05 USD/EUR -> Speso ~95 EUR
                # Nota: Usiamo il cambio attuale per semplicità, l'App usa quello storico.
                # Ma per avere una stima va benissimo.
                cost_in_eur = cost / current_eur_usd_rate

            # Aggiorna il sommario per l'asset base
            if base_asset:
                if base_asset not in summary:
                    summary[base_asset] = {'total_cost_eur': 0.0, 'total_vol': 0.0}

                summary[base_asset]['total_cost_eur'] += cost_in_eur
                summary[base_asset]['total_vol'] += vol

    return summary

def get_open_orders():
    try:
        # 3. Esegui la richiesta privata 'OpenOrders'
        # Il parametro 'trades': True include i dettagli se l'ordine è stato parzialmente eseguito
        response = k.query_private('OpenOrders', {'trades': True})

        # 4. Gestione degli errori
        if response.get('error'):
            print(f"Errore API Kraken: {response['error']}")
            return

        # 5. Estrai i risultati
        result = response.get('result', {})
        open_orders = result.get('open', {})

        if not open_orders:
            print("Non ci sono ordini aperti al momento.")
            return

        print(f"Trovati {len(open_orders)} ordini aperti:\n")

        # 6. Itera e stampa i dettagli degli ordini
        for txid, order_info in open_orders.items():
            pair = order_info.get('descr', {}).get('pair') # Es. XBTEUR
            type_op = order_info.get('descr', {}).get('type') # buy o sell
            price = order_info.get('descr', {}).get('price') # Prezzo impostato
            vol = order_info.get('vol') # Volume totale
            vol_exec = order_info.get('vol_exec') # Volume già eseguito (se parziale)
            status = order_info.get('status') # pending, open, ecc.

            print(f"ID Ordine: {txid}")
            print(f" - Coppia: {pair}")
            print(f" - Tipo: {type_op} (Prezzo: {price})")
            print(f" - Stato: {status}")
            print(f" - Volume: {vol_exec} / {vol} eseguito")
            print("-" * 30)

    except Exception as e:
        print(f"Si è verificato un errore generico: {e}")

def get_open_positions():
    try:
        # 'docalcs': True è importante: calcola il Profitto/Perdita (net) attuale
        response = k.query_private('OpenPositions', {'docalcs': True})

        if response.get('error'):
            print(f"Errore API: {response['error']}")
            return

        result = response.get('result', {})

        # Se result è vuoto o non ci sono chiavi, non hai posizioni di margine aperte
        if not result:
            print("Nessuna posizione di margine aperta.")
            return

        print(f"Trovate {len(result)} posizioni aperte:\n")

        for pos_id, data in result.items():
            pair = data.get('pair')       # Es. XBTUSD
            type_pos = data.get('type')   # buy (Long) o sell (Short)
            vol = float(data.get('vol'))  # Quantità
            cost = float(data.get('cost')) # Costo base della posizione

            # 'net' è il P&L (Profitto/Perdita) incluso di fee, disponibile grazie a docalcs=True
            pnl = float(data.get('net'))

            # Prezzo medio di entrata (Costo / Volume)
            entry_price = cost / vol if vol else 0

            # Valore corrente della posizione
            current_value = float(data.get('value'))

            print(f"ID Posizione: {pos_id}")
            print(f" - Coppia: {pair} ({type_pos.upper()})")
            print(f" - Quantità: {vol}")
            print(f" - Prezzo Entrata: {entry_price:.5f}")
            print(f" - Valore Attuale: {current_value:.2f}")

            # Colora (simbolicamente) il P&L
            sign = "+" if pnl >= 0 else ""
            print(f" - P&L (Profitto/Perdita): {sign}{pnl:.2f}")
            print("-" * 30)

    except Exception as e:
        print(f"Errore generico: {e}")


def get_normalized_portfolio():
    print("--- Analisi Portafoglio (Spot) ---")

    try:
        # 1. Recupero Dati Attuali
        ticker_resp = k.query_public('Ticker', {'pair': 'EURUSD'})
        eur_usd_rate = 1.05 # Default
        if not ticker_resp.get('error'):
            pair_data = ticker_resp['result'].get('ZEURZUSD') or ticker_resp['result'].get('EURUSD')
            if pair_data:
                eur_usd_rate = float(pair_data['c'][0])

        # 2. Scarichiamo storico e bilancio
        trades_summary = get_trades_history_summary(current_eur_usd_rate=eur_usd_rate)

        assets_resp = k.query_public('Assets')
        pairs_resp = k.query_public('AssetPairs')
        balance_resp = k.query_private('Balance')

        if assets_resp.get('error'): raise Exception(assets_resp['error'])
        if balance_resp.get('error'): raise Exception(balance_resp['error'])

        my_balance = balance_resp['result']
        public_pairs = pairs_resp['result']
        public_assets = assets_resp['result']

        asset_to_eur_pair = {}
        for p, info in public_pairs.items():
            if info.get('quote') == 'ZEUR':
                asset_to_eur_pair[info.get('base')] = p

        # 3. Costruzione Portafoglio
        portfolio_items = []
        pairs_to_fetch = set()
        usd_amount = float(my_balance.get('ZUSD', 0))

        for asset_id, amount in my_balance.items():
            amount = float(amount)
            if amount <= 0 or asset_id == 'ZEUR' or asset_id == 'ZUSD': continue

            # A. Mappatura Asset
            lookup_id = HISTORY_MAPPING.get(asset_id, asset_id)

            # Pulizia suffissi
            if lookup_id not in asset_to_eur_pair and '.' in lookup_id:
                clean = lookup_id.split('.')[0]
                if clean in asset_to_eur_pair: lookup_id = clean
                elif ('X'+clean) in asset_to_eur_pair: lookup_id = 'X'+clean

            # B. Recupero Prezzo Medio
            avg_price = 0.0
            hist_data = trades_summary.get(lookup_id)

            if not hist_data and lookup_id.startswith('X') and len(lookup_id) == 4:
                 hist_data = trades_summary.get(lookup_id[1:])

            if hist_data and hist_data['total_vol'] > 0:
                avg_price = hist_data['total_cost_eur'] / hist_data['total_vol']

            # C. Identifica Coppia
            pair_name = asset_to_eur_pair.get(lookup_id)
            display_name = DISPLAY_MAP.get(asset_id, public_assets.get(asset_id, {}).get('altname', asset_id))

            if pair_name:
                portfolio_items.append({
                    'name': display_name,
                    'pair': pair_name,
                    'amount': amount,
                    'entry_price': avg_price
                })
                pairs_to_fetch.add(pair_name)
            else:
                 print(f" [Skip] {display_name}: No pair EUR")

        if usd_amount > 0: pairs_to_fetch.add('EURUSD')

        # 4. Recupero Prezzi Attuali
        if not pairs_to_fetch: return

        t_resp = k.query_public('Ticker', {'pair': ",".join(list(pairs_to_fetch))})
        if t_resp.get('error'): return
        tickers = t_resp['result']

        total_val = float(my_balance.get('ZEUR', 0))

        print("\n" + "="*65)
        print(f"{'ASSET':<18} {'QTA':<10} {'ENTRY (€)':<12} {'ATTUALE (€)':<12} {'P&L (€)':<10}")
        print("-" * 65)

        for item in portfolio_items:
            pair = item['pair']
            data = tickers.get(pair)

            # Fallback ticker
            if not data:
                # --- CORREZIONE QUI SOTTO: Ho cambiato 'k' in 'ticker_key' ---
                for ticker_key, v in tickers.items():
                    if pair in ticker_key:
                        data = v
                        break

            if data:
                price = float(data['c'][0])
                val = item['amount'] * price
                total_val += val

                entry_str = "N/A"
                pnl_str = "-"

                if item['entry_price'] > 0:
                    entry = item['entry_price']
                    cost_basis = item['amount'] * entry
                    diff = val - cost_basis
                    sign = "+" if diff >= 0 else ""
                    pnl_str = f"{sign}{diff:.2f}"
                    entry_str = f"{entry:.2f}"
                else:
                    entry_str = "(No Hist)"

                print(f"{item['name']:<18} {item['amount']:<10.5f} {entry_str:<12} {val:<12.2f} {pnl_str:<10}")

        # USD
        if usd_amount > 0:
            d_data = tickers.get('EURUSD') or tickers.get('ZEURZUSD')
            if d_data:
                rate = float(d_data['c'][0])
                val_eur = usd_amount * (1/rate)
                total_val += val_eur
                print(f"{'USD Cash':<18} {usd_amount:<10.2f} {'(Fiat)':<12} {val_eur:<12.2f} {'-':<10}")

        liq = float(my_balance.get('ZEUR', 0))
        if liq > 0:
            print("-" * 65)
            print(f"LIQUIDITÀ: {liq:.2f} €")

        print("="*65)
        print(f"TOTALE ESTIMATO: {total_val:.2f} €\n")

    except Exception as e:
        print(f"Errore: {e}")

if __name__ == "__main__":
    get_open_orders()
    get_open_positions()
    get_normalized_portfolio()
