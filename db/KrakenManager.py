import os
import krakenex
from dotenv import load_dotenv

class KrakenPortfolioManager:
    def __init__(self):
        # 1. Caricamento configurazione
        load_dotenv()
        self.api_key = os.getenv('KRAKEN_KEY')
        self.api_secret = os.getenv('KRAKEN_SECRET')

        # 2. Inizializzazione API
        self.k = krakenex.API(key=self.api_key, secret=self.api_secret)

        # 3. Lista accumulatore e Cache saldo
        self.positions = []
        self.balance_info = None # Qui salveremo la risposta di TradeBalance

        # 4. Cache Dati Pubblici (Assets e Pairs) per conversioni nomi veloci
        print("--- Inizializzazione: Scaricamento info Asset e Coppie ---")
        self.public_assets = {}
        self.public_pairs = {}
        self._cache_public_data()

        # Mappe manuali per display più pulito
        self.DISPLAY_MAP = {
            'XXBT': 'BTC', 'XBT.B': 'BTC', 'XETH': 'ETH', 'ETH2': 'ETH',
            'XDOT': 'DOT', 'XLTC': 'LTC', 'ZEUR': 'EUR', 'ZUSD': 'USD'
        }
        self.HISTORY_MAPPING = {'XBT.B': 'XXBT', 'ETH2': 'XETH', 'ETH2.S': 'XETH'}

    def _cache_public_data(self):
        try:
            a_resp = self.k.query_public('Assets')
            p_resp = self.k.query_public('AssetPairs')

            if not a_resp.get('error') and not p_resp.get('error'):
                self.public_assets = a_resp['result']
                self.public_pairs = p_resp['result']
        except Exception as e:
            print(f"Errore cache dati pubblici: {e}")

    def _create_empty_record(self):
        """Genera il template vuoto per uniformare l'output"""
        return {
            'pair': None,          # Human: BTC/EUR
            'kr_pair': None,       # Kraken: XXBTZEUR
            'base': None,          # BTC
            'quote': None,         # EUR
            'qty': 0.0,
            'price_entry': None,   # Per Margin o Ordini
            'price_avg': None,     # Per Spot (Media storica)
            'take_profit': None,
            'stop_loss': None,
            'price': None,         # Prezzo Attuale Mercato
            'value_eur': 0.0,      # <--- NUOVO CAMPO: Valore in Euro pre-calcolato
            'pnl': None,
            'type': None,          # 'order', 'position_margin', 'position'
            'subtype': None        # 'buy', 'sell'
        }

    def _get_pair_details(self, kraken_pair_name):
        info = self.public_pairs.get(kraken_pair_name, {})
        if not info:
            return kraken_pair_name, kraken_pair_name, "N/A", "N/A"

        wsname = info.get('wsname', kraken_pair_name)
        base_id = info.get('base')
        quote_id = info.get('quote')

        base_human = self.DISPLAY_MAP.get(base_id, self.public_assets.get(base_id, {}).get('altname', base_id))
        quote_human = self.DISPLAY_MAP.get(quote_id, self.public_assets.get(quote_id, {}).get('altname', quote_id))

        return wsname, base_human, quote_human

    def _get_ticker_price(self, pairs_list):
        if not pairs_list: return {}
        try:
            resp = self.k.query_public('Ticker', {'pair': ",".join(list(set(pairs_list)))})
            if not resp.get('error'):
                return resp['result']
        except:
            pass
        return {}

    # =========================================================================
    # METODO 1: ORDINI APERTI
    # =========================================================================
    def get_open_orders(self):
        print("--- Recupero Ordini Aperti ---")
        try:
            response = self.k.query_private('OpenOrders', {'trades': True})
            if response.get('error'): return

            open_orders = response.get('result', {}).get('open', {})
            pairs_to_fetch = [o['descr']['pair'] for o in open_orders.values()]
            tickers = self._get_ticker_price(pairs_to_fetch)

            for txid, order in open_orders.items():
                record = self._create_empty_record()
                descr = order['descr']
                kr_pair = descr['pair']

                pair_human, base, quote = self._get_pair_details(kr_pair)

                record['type'] = 'order'
                record['pair'] = pair_human
                record['kr_pair'] = kr_pair
                record['base'] = base
                record['quote'] = quote
                record['subtype'] = descr['type']

                vol = float(order.get('vol', 0))
                vol_exec = float(order.get('vol_exec', 0))
                record['qty'] = vol - vol_exec
                record['price_entry'] = float(descr['price'])

                if 'stopprice' in order:
                     record['stop_loss'] = float(order['stopprice'])

                ticker_data = tickers.get(kr_pair) or tickers.get(pair_human)
                if ticker_data:
                    record['price'] = float(ticker_data['c'][0])
                    # Per gli ordini non calcoliamo value_eur finché non sono eseguiti

                self.positions.append(record)

        except Exception as e:
            print(f"Eccezione OpenOrders: {e}")

    # =========================================================================
    # METODO 2: POSIZIONI MARGIN
    # =========================================================================
    def get_open_positions(self):
        print("--- Recupero Posizioni Margin ---")
        try:
            response = self.k.query_private('OpenPositions', {'docalcs': True})
            if response.get('error'): return

            positions = response.get('result', {})

            for pos_id, data in positions.items():
                record = self._create_empty_record()

                kr_pair = data['pair']
                pair_human, base, quote = self._get_pair_details(kr_pair)

                record['type'] = 'position_margin'
                record['pair'] = pair_human
                record['kr_pair'] = kr_pair
                record['base'] = base
                record['quote'] = quote
                record['subtype'] = data['type']

                vol = float(data['vol'])
                cost = float(data['cost'])
                value = float(data['value']) # Valore corrente secondo Kraken

                record['qty'] = vol
                record['price_entry'] = cost / vol if vol else 0
                record['pnl'] = float(data['net'])
                record['price'] = value / vol if vol else 0

                # --- MODIFICA: Salviamo il valore ---
                record['value_eur'] = value

                self.positions.append(record)

        except Exception as e:
            print(f"Eccezione OpenPositions: {e}")

    # =========================================================================
    # METODO 3: PORTAFOGLIO SPOT
    # =========================================================================
    def _get_trades_history(self, eur_usd_rate=1.05):
        all_trades = {}
        offset = 0
        while True:
            try:
                resp = self.k.query_private('TradesHistory', {'type': 'no position', 'limit': 1000, 'ofs': offset})
                if resp.get('error'): break
                res = resp['result']
                trades = res.get('trades', {})
                if not trades: break
                all_trades.update(trades)
                offset = res.get('last')
                if len(trades) < 1000: break
            except: break

        summary = {}
        for info in all_trades.values():
            if info['type'] == 'buy':
                pair = info['pair']
                cost = float(info['cost'])
                vol = float(info['vol'])

                base = None
                if 'XBT' in pair or 'BTC' in pair: base = 'XXBT'
                elif 'ETH' in pair: base = 'XETH'
                elif 'SOL' in pair: base = 'SOL'
                elif 'DOT' in pair: base = 'XDOT'
                elif 'ADA' in pair: base = 'ADA'
                else:
                     base = pair[:4] if pair.startswith('X') and len(pair)>=8 else pair[:3]

                cost_eur = cost
                if 'USD' in pair: cost_eur = cost / eur_usd_rate

                if base:
                    if base not in summary: summary[base] = {'cost': 0.0, 'vol': 0.0}
                    summary[base]['cost'] += cost_eur
                    summary[base]['vol'] += vol
        return summary

    def get_normalized_portfolio(self):
        print("--- Recupero Portafoglio Spot ---")
        try:
            ticker_usd = self.k.query_public('Ticker', {'pair': 'EURUSD'})
            eur_rate = 1.05
            if not ticker_usd.get('error'):
                d = ticker_usd['result'].get('ZEURZUSD') or ticker_usd['result'].get('EURUSD')
                if d: eur_rate = float(d['c'][0])

            history = self._get_trades_history(eur_rate)
            bal_resp = self.k.query_private('Balance')
            if bal_resp.get('error'): return
            balance = bal_resp['result']

            asset_to_eur_pair = {}
            for p_name, info in self.public_pairs.items():
                if info.get('quote') == 'ZEUR':
                    asset_to_eur_pair[info.get('base')] = p_name

            pairs_to_fetch = []
            temp_items = []

            for asset, amount in balance.items():
                amount = float(amount)
                if amount <= 0 or asset in ['ZEUR', 'ZUSD', 'KFEE']: continue

                record = self._create_empty_record()
                record['type'] = 'position'
                record['subtype'] = 'buy'
                record['qty'] = amount

                lookup_id = self.HISTORY_MAPPING.get(asset, asset)
                if lookup_id not in asset_to_eur_pair and '.' in lookup_id:
                    clean = lookup_id.split('.')[0]
                    if clean in asset_to_eur_pair: lookup_id = clean
                    elif ('X'+clean) in asset_to_eur_pair: lookup_id = 'X'+clean

                hist_data = history.get(lookup_id)
                if not hist_data and lookup_id.startswith('X') and len(lookup_id)==4:
                    hist_data = history.get(lookup_id[1:])

                if hist_data and hist_data['vol'] > 0:
                    record['price_avg'] = hist_data['cost'] / hist_data['vol']

                kr_pair = asset_to_eur_pair.get(lookup_id)
                if kr_pair:
                    pairs_to_fetch.append(kr_pair)
                    pair_human, base, quote = self._get_pair_details(kr_pair)
                    record['pair'] = pair_human
                    record['kr_pair'] = kr_pair
                    record['base'] = base
                    record['quote'] = quote
                else:
                    record['base'] = self.DISPLAY_MAP.get(asset, asset)
                    record['pair'] = f"{record['base']}/?"

                temp_items.append(record)

            usd_bal = float(balance.get('ZUSD', 0))
            if usd_bal > 0:
                pairs_to_fetch.append('EURUSD')
                rec_usd = self._create_empty_record()
                rec_usd['type'] = 'position'
                rec_usd['subtype'] = 'buy'
                rec_usd['base'] = 'USD'
                rec_usd['quote'] = 'EUR'
                rec_usd['qty'] = usd_bal
                rec_usd['pair'] = 'USD/EUR'
                rec_usd['kr_pair'] = 'EURUSD'
                temp_items.append(rec_usd)

            tickers = self._get_ticker_price(pairs_to_fetch)

            for item in temp_items:
                kr_p = item['kr_pair']
                t_data = tickers.get(kr_p)
                if not t_data:
                    for k_t, v_t in tickers.items():
                        if kr_p and kr_p in k_t:
                            t_data = v_t
                            break

                if t_data:
                    current_price = float(t_data['c'][0])

                    if item['base'] == 'USD':
                        current_price = 1 / current_price

                    item['price'] = current_price

                    # --- MODIFICA: Salviamo il valore qui per evitare ricalcoli ---
                    val_in_eur = item['qty'] * current_price
                    item['value_eur'] = val_in_eur

                    if item['price_avg']:
                        item['pnl'] = val_in_eur - (item['price_avg'] * item['qty'])

                self.positions.append(item)

        except Exception as e:
            print(f"Eccezione NormalizedPortfolio: {e}")

    # =========================================================================
    # METODO 4: RIEPILOGO TOTALE (NUOVO)
    # =========================================================================
    def get_portfolio_summary(self):
        """
        Calcola i totali del portafoglio.
        Fa UNA chiamata API 'TradeBalance' per avere Equity e Free Margin precisi.
        Il resto è aggregato da self.positions.
        """
        print("--- Calcolo Riepilogo ---")

        # 1. Recupero Equity ufficiale e Free Margin (TradeBalance)
        equity_official = 0.0
        free_margin = 0.0

        try:
            # asset='ZEUR' chiede il controvalore totale in Euro
            tb_resp = self.k.query_private('TradeBalance', {'asset': 'ZEUR'})
            if not tb_resp.get('error'):
                res = tb_resp['result']
                equity_official = float(res['eb']) # Equivalent Balance (Equity)
                free_margin = float(res['mf'])     # Margin Free (Disponibile)
                self.balance_info = res            # Salviamo per usi futuri
        except Exception as e:
            print(f"Errore TradeBalance: {e}")

        # 2. Aggregazione manuale P&L e Valore Lordo Assets
        total_pnl = 0.0
        total_gross_assets = 0.0 # Valore di tutte le posizioni (Spot + Margin)

        for p in self.positions:
            # Somma PnL (se esiste)
            if p['pnl'] is not None:
                total_pnl += p['pnl']

            # Somma Valore Euro degli asset (se non è un ordine pendente)
            if p['type'] != 'order':
                total_gross_assets += p.get('value_eur', 0.0)

        # Fallback se API fallisce
        if equity_official == 0.0:
            equity_official = total_gross_assets

        return {
            "total_equity_stimata": equity_official,      # Equity Totale (Cash + Crypto + PnL)
            "pnl": total_pnl,                             # PnL Totale (Spot + Margin)
            "totale_portafoglio": total_gross_assets,     # Valore lordo degli asset investiti
            "totale_portafoglio_disponibile": free_margin, # Liquidità usabile per nuovi trade
            "totale_portafoglio_liquido": equity_official - total_gross_assets, # Liquidità usabile per nuovi trade
        }

    # =========================================================================
    # UTILITY RETURN
    # =========================================================================
    def get_all_positions_data(self):
        return self.positions

# =============================================================================
# FUNZIONE DI STAMPA ESTERNA
# =============================================================================
def print_pretty_report(positions):
    # Colori ANSI per il terminale
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

    print(f"\n{BOLD}{'='*110}{RESET}")
    print(f"{BOLD}{'TIPO':<10} {'COPPIA':<12} {'LATO':<6} {'QTA':<12} {'ENTRY (€)':<12} {'PREZZO (€)':<12} {'VALORE (€)':<12} {'P&L (€)':<12}{RESET}")
    print(f"{BOLD}{'='*110}{RESET}")

    total_equity = 0.0
    total_pnl = 0.0

    # Ordiniamo per tipo per raggrupparli visivamente
    sorted_positions = sorted(positions, key=lambda x: x['type'], reverse=True)

    for p in sorted_positions:
        p_type = p.get('type', '').replace('position_', '').upper()
        if p_type == 'POSITION': p_type = 'SPOT'

        pair = p.get('pair') or 'N/A'
        side = (p.get('subtype') or '').upper()
        qty = p.get('qty', 0.0)
        curr_price = p.get('price')

        price_str = "N/A"
        value_str = "N/A"

        # Usiamo il campo value_eur già calcolato se disponibile
        value = p.get('value_eur', 0.0)

        if curr_price:
            price_str = f"{curr_price:.5f}"
            value_str = f"{value:.2f}"
            if p_type != 'ORDER':
                total_equity += value

        entry = p.get('price_entry') if p.get('price_entry') else p.get('price_avg')
        entry_str = f"{entry:.5f}" if entry else "-"

        pnl = p.get('pnl')
        pnl_str = "-"

        if pnl is not None:
            color = GREEN if pnl >= 0 else RED
            sign = "+" if pnl >= 0 else ""
            pnl_str = f"{color}{sign}{pnl:.2f}{RESET}"
            total_pnl += pnl
        elif p_type == 'SPOT' and entry is None:
            pnl_str = f"{YELLOW}(No Hist){RESET}"

        side_color = GREEN if side == 'BUY' else RED
        side_fmt = f"{side_color}{side}{RESET}"

        if p.get('base') == 'USD': price_str = "(Fiat)"

        print(f"{p_type:<10} {pair:<12} {side_fmt:<15} {qty:<12.6f} {entry_str:<12} {price_str:<12} {value_str:<12} {pnl_str:<12}")

    print(f"{'-'*110}")
    print(f"{BOLD}TOTALE EQUITY ESTIMATA:{RESET} {total_equity:.2f} €")
    print(f"{BOLD}TOTALE P&L (Real time):{RESET} {GREEN if total_pnl>=0 else RED}{total_pnl:+.2f} €{RESET}")
    print(f"{'='*110}\n")

if __name__ == "__main__":
    manager = KrakenPortfolioManager()

    # 1. Popoliamo i dati
    manager.get_open_orders()
    manager.get_open_positions()
    manager.get_normalized_portfolio()

    # 2. Stampa report dettagliato
    all_data = manager.get_all_positions_data()
    print_pretty_report(all_data)

    # 3. Ottieni oggetto riepilogo
    summary = manager.get_portfolio_summary()
    print("\n=== OGGETTO SUMMARY PER IL BOT ===")
    print(summary)
