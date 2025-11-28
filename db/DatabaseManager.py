# --- START OF FILE DatabaseManager.py ---

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

    def insert_currency_data(self, candles_list, pair_info, table_name: str = "currency"):
            if not candles_list:
                return

            safe_table = "".join(ch for ch in table_name if ch.isalnum() or ch == '_')
            if not safe_table:
                print(f"[ERROR] Nome tabella non valido: {table_name}")
                return

            # Recuperiamo i dati base
            p_pair = pair_info.get('pair')
            p_kr = pair_info.get('kr_pair')
            p_base = pair_info.get('base')
            p_quote = pair_info.get('quote')

            # Controllo di sicurezza
            if not p_pair:
                print(f"[ERROR] Coppia senza nome trovata.")
                return

            # MODIFICA: Uso ON DUPLICATE KEY UPDATE invece di INSERT IGNORE
            # Questo evita duplicati se esiste un indice UNIQUE(pair, timeframe, timestamp)
            query = f"""
                INSERT INTO {safe_table} (
                    pair, kr_pair, base, quote,
                    timestamp, open, high, low, close, volume,
                    bid, ask, mid, spread,
                    ema_fast, ema_slow, rsi, atr, timeframe,
                    created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                ON DUPLICATE KEY UPDATE
                    open = VALUES(open),
                    high = VALUES(high),
                    low = VALUES(low),
                    close = VALUES(close),
                    volume = VALUES(volume),
                    ema_fast = VALUES(ema_fast),
                    ema_slow = VALUES(ema_slow),
                    rsi = VALUES(rsi),
                    atr = VALUES(atr),
                    created_at = NOW()
            """

            data_tuples = []

            for c in candles_list:
                # Tupla: Deve avere esattamente 19 valori
                row = (
                    p_pair,         # 1. pair (che sostituisce il vecchio ID e il vecchio pair)
                    p_kr,           # 2. kr_pair
                    p_base,         # 3. base
                    p_quote,        # 4. quote
                    c.get('timestamp'),
                    c.get('open'),
                    c.get('high'),
                    c.get('low'),
                    c.get('close'),
                    c.get('volume'),
                    c.get('bid'),
                    c.get('ask'),
                    c.get('mid'),
                    c.get('spread'),
                    c.get('ema_fast'),
                    c.get('ema_slow'),
                    c.get('rsi'),
                    c.get('atr'),
                    c.get('timeframe')   # 19. timeframe
                )
                data_tuples.append(row)

            try:
                self.cursor.executemany(query, data_tuples)
                self.conn.commit()
                print(f" -> Inserite/Aggiornate {self.cursor.rowcount} righe per {p_pair}")
            except mysql.connector.Error as err:
                print(f"Errore insert_currency_data: {err}")
                self.conn.rollback()

    # =====================================================
    # NUOVO METODO: INSERT ALL PAIRS (SETUP)
    # =====================================================
    def insert_all_pairs(self, all_pairs_list):
        """
        Popola o aggiorna la tabella pair_limits usando tutti i campi restituiti da getAllPairs.
        Usa ON DUPLICATE KEY UPDATE per aggiornare i record esistenti.
        """
        if not all_pairs_list:
            print(" -> Nessuna coppia da inserire.")
            return

        query = """
            INSERT INTO pair_limits (
                pair, kr_pair, base, quote,
                lot_decimals, ordermin, pair_decimals,
                fee_volume_currency, fees, fees_maker,
                leverage_buy, leverage_sell,
                leverage_buy_max, leverage_sell_max,
                can_leverage_buy, can_leverage_sell
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                kr_pair = VALUES(kr_pair),
                base = VALUES(base),
                quote = VALUES(quote),
                lot_decimals = VALUES(lot_decimals),
                ordermin = VALUES(ordermin),
                pair_decimals = VALUES(pair_decimals),
                fee_volume_currency = VALUES(fee_volume_currency),
                fees = VALUES(fees),
                fees_maker = VALUES(fees_maker),
                leverage_buy = VALUES(leverage_buy),
                leverage_sell = VALUES(leverage_sell),
                leverage_buy_max = VALUES(leverage_buy_max),
                leverage_sell_max = VALUES(leverage_sell_max),
                can_leverage_buy = VALUES(can_leverage_buy),
                can_leverage_sell = VALUES(can_leverage_sell)
        """

        data_tuples = []
        for p in all_pairs_list:
            limits = p.get('pair_limits', {}) or {}
            row = (
                p.get('pair'),
                p.get('kr_pair'),
                p.get('base'),
                p.get('quote'),
                limits.get('lot_decimals'),
                limits.get('ordermin'),
                limits.get('pair_decimals'),
                limits.get('fee_volume_currency'),
                json.dumps(limits.get('fees', [])),
                json.dumps(limits.get('fees_maker', [])),
                json.dumps(limits.get('leverage_buy', [])),
                json.dumps(limits.get('leverage_sell', [])),
                limits.get('leverage_buy_max'),
                limits.get('leverage_sell_max'),
                1 if limits.get('can_leverage_buy') else 0,
                1 if limits.get('can_leverage_sell') else 0
            )
            data_tuples.append(row)

        try:
            self.cursor.executemany(query, data_tuples)
            self.conn.commit()
            print(f" -> Setup completato: Inserite/Aggiornate {len(data_tuples)} coppie in pair_limits.")
        except mysql.connector.Error as err:
            print(f"Errore insert_all_pairs: {err}")
            self.conn.rollback()

    def close_connection(self):
        if self.cursor: self.cursor.close()
        if self.conn: self.conn.close()
        print("--- Connessione Database Chiusa ---")

    def select_all(self, table_name: str, where_clause: str = "1"):
        """
        Esegue una SELECT * su una tabella a scelta usando una clausola WHERE testuale.
        Restituisce una lista di dizionari {colonna: valore}.
        """
        safe_table = "".join(ch for ch in table_name if ch.isalnum() or ch == '_')
        if not safe_table:
            print(f"[ERROR] Nome tabella non valido: {table_name}")
            return []

        clause = where_clause.strip() if where_clause else "1"
        query = f"SELECT * FROM {safe_table} WHERE {clause}"

        try:
            self.cursor.execute(query)
            rows = self.cursor.fetchall()
            columns = [col[0] for col in self.cursor.description] if self.cursor.description else []
            return [dict(zip(columns, row)) for row in rows]
        except mysql.connector.Error as err:
            print(f"Errore select_all su {safe_table}: {err}")
            return []
