import os
import json
import mysql.connector
from dotenv import load_dotenv

class DatabaseManager:
    def __init__(self):
        load_dotenv()
        try:
            self.conn = mysql.connector.connect(
                host=os.getenv('MYSQL_HOST'),
                user=os.getenv('MYSQL_USER', 'root'),
                password=os.getenv('MYSQL_PASSWORD'),
                database=os.getenv('MYSQL_DATABASE'),
                port=int(os.getenv('MYSQL_PORT', 3306))
            )
            self.cursor = self.conn.cursor()
            print("--- Database: Connessione stabilita con successo ---")
        except mysql.connector.Error as err:
            print(f"Errore di connessione al Database: {err}")
            raise err

    def _get_pair_limit_id(self, pair_name):
        query = "SELECT id FROM pair_limits WHERE pair = %s LIMIT 1"
        try:
            self.cursor.execute(query, (pair_name,))
            result = self.cursor.fetchone()
            if result: return result[0]
            return None
        except mysql.connector.Error:
            return None

    def insert_wallet(self, summary_data):
        query = """
            INSERT INTO wallet (
                total_equity_stimata, pnl, totale_portafoglio,
                totale_portafoglio_disponibile, totale_portafoglio_liquido,
                created_at, record_date
            ) VALUES (%s, %s, %s, %s, %s, NOW(), CURDATE())
        """
        values = (
            summary_data.get('total_equity_stimata'),
            summary_data.get('pnl'),
            summary_data.get('totale_portafoglio'),
            summary_data.get('totale_portafoglio_disponibile'),
            summary_data.get('totale_portafoglio_liquido', 0.0)
        )
        try:
            self.cursor.execute(query, values)
            self.conn.commit()
            return self.cursor.lastrowid
        except mysql.connector.Error as err:
            print(f"Errore insert_wallet: {err}")
            self.conn.rollback()
            return None

    def insert_orders(self, wallet_id, positions_list):
        if not positions_list: return None
        query = """
            INSERT INTO orders (
                wallet_id, pair, kr_pair, base, quote, qty,
                price_entry, price_avg, take_profit, stop_loss,
                price, value_eur, pnl, type, subtype,
                created_at, record_date
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), CURDATE())
        """
        data_tuples = []
        for p in positions_list:
            row = (
                wallet_id, p.get('pair'), p.get('kr_pair'), p.get('base'), p.get('quote'),
                p.get('qty'), p.get('price_entry'), p.get('price_avg'),
                p.get('take_profit'), p.get('stop_loss'), p.get('price'),
                p.get('value_eur'), p.get('pnl'), p.get('type'), p.get('subtype')
            )
            data_tuples.append(row)
        try:
            self.cursor.executemany(query, data_tuples)
            self.conn.commit()
            return self.cursor.lastrowid
        except mysql.connector.Error as err:
            print(f"Errore insert_orders: {err}")
            self.conn.rollback()
            return None

    def insert_currency_data(self, candles_list, pair_info):
        if not candles_list: return
        pair_name = pair_info.get('pair')
        pair_limits_id = self._get_pair_limit_id(pair_name)

        if not pair_limits_id:
            print(f"[ERROR] Coppia '{pair_name}' non trovata in pair_limits.")
            return

        query = """
            INSERT IGNORE INTO currency (
                pair_limits_id, pair, kr_pair, base, quote,
                timestamp, open, high, low, close, volume,
                bid, ask, mid, spread, ema_fast, ema_slow,
                created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
        """
        data_tuples = []
        p_pair = pair_info.get('pair')
        p_kr = pair_info.get('kr_pair')
        p_base = pair_info.get('base')
        p_quote = pair_info.get('quote')

        for c in candles_list:
            row = (
                pair_limits_id, p_pair, p_kr, p_base, p_quote,
                c.get('timestamp'), c.get('open'), c.get('high'), c.get('low'), c.get('close'),
                c.get('volume'), c.get('bid'), c.get('ask'), c.get('mid'), c.get('spread'),
                c.get('ema_fast'), c.get('ema_slow')
            )
            data_tuples.append(row)
        try:
            self.cursor.executemany(query, data_tuples)
            self.conn.commit()
            print(f" -> Inserite/Aggiornate {self.cursor.rowcount} candele per {p_pair}")
        except mysql.connector.Error as err:
            print(f"Errore insert_currency_data: {err}")
            self.conn.rollback()

    # =====================================================
    # NUOVO METODO: INSERT ALL PAIRS (SETUP)
    # =====================================================
    def insert_all_pairs(self, all_pairs_list):
        """
        Popola o AGGIORNA la tabella pair_limits.
        Usa ON DUPLICATE KEY UPDATE per forzare il salvataggio dei dati.
        """
        if not all_pairs_list:
            print(" -> Nessuna coppia da inserire.")
            return

        # Query UPSERT (Insert o Update se esiste già)
        query = """
            INSERT INTO pair_limits (
                pair, lot_decimals, ordermin, pair_decimals,
                fee_volume_currency, leverage_buy, leverage_sell,
                leverage_buy_max, leverage_sell_max,
                can_leverage_buy, can_leverage_sell
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                lot_decimals = VALUES(lot_decimals),
                ordermin = VALUES(ordermin),
                pair_decimals = VALUES(pair_decimals),
                fee_volume_currency = VALUES(fee_volume_currency),
                leverage_buy = VALUES(leverage_buy),
                leverage_sell = VALUES(leverage_sell),
                leverage_buy_max = VALUES(leverage_buy_max),
                leverage_sell_max = VALUES(leverage_sell_max),
                can_leverage_buy = VALUES(can_leverage_buy),
                can_leverage_sell = VALUES(can_leverage_sell)
        """

        data_tuples = []
        for p in all_pairs_list:
            limits = p.get('pair_limits', {})

            lev_buy_json = json.dumps(limits.get('leverage_buy', []))
            lev_sell_json = json.dumps(limits.get('leverage_sell', []))

            can_buy = 1 if limits.get('can_leverage_buy') else 0
            can_sell = 1 if limits.get('can_leverage_sell') else 0

            row = (
                p.get('pair'),
                limits.get('lot_decimals'),
                limits.get('ordermin'),
                limits.get('pair_decimals'),
                limits.get('fee_volume_currency'),
                lev_buy_json,
                lev_sell_json,
                limits.get('leverage_buy_max'),
                limits.get('leverage_sell_max'),
                can_buy,
                can_sell
            )
            data_tuples.append(row)

        try:
            self.cursor.executemany(query, data_tuples)
            self.conn.commit() # Importante: Conferma la transazione
            print(f" -> Setup completato: Inserite/Aggiornate {len(data_tuples)} coppie in pair_limits.")
        except mysql.connector.Error as err:
            print(f"Errore insert_all_pairs: {err}")
            self.conn.rollback()

    def close_connection(self):
        if self.cursor: self.cursor.close()
        if self.conn: self.conn.close()
        print("--- Connessione Database Chiusa ---")
