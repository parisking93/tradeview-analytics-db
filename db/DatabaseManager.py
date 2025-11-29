# --- START OF FILE DatabaseManager.py ---

import os
import json
import mysql.connector
from dotenv import load_dotenv
from datetime import datetime, timedelta

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

    def get_candles_with_offset(self, table_name: str, timeframe: str, base: str, offset: int):
        """
        Restituisce le righe ordinate per timestamp desc con offset personalizzato.
        Esempio query:
        SELECT * FROM currency
        WHERE timeframe = '1d' AND base = 'ETH'
        ORDER BY STR_TO_DATE(`timestamp`, '%Y-%m-%d %H:%i:%s') DESC
        LIMIT 18446744073709551615 OFFSET 118;
        """
        safe_table = "".join(ch for ch in table_name if ch.isalnum() or ch == '_')
        if not safe_table:
            print(f"[ERROR] Nome tabella non valido: {table_name}")
            return []

        query = f"""
            SELECT *
            FROM {safe_table}
            WHERE timeframe = %s AND base = %s
            ORDER BY CAST(`timestamp` AS DATETIME) DESC
            LIMIT 18446744073709551615 OFFSET %s
        """
        try:
            self.cursor.execute(query, (timeframe, base, int(offset)))
            rows = self.cursor.fetchall()
            columns = [col[0] for col in self.cursor.description] if self.cursor.description else []
            return [dict(zip(columns, row)) for row in rows]
        except mysql.connector.Error as err:
            print(f"Errore get_candles_with_offset su {safe_table}: {err}")
            return []


    def get_candles_before_date(self, table_name: str, timeframe: str, base: str, cutoff_datetime):
        """
        Restituisce le righe per timeframe/base con timestamp < cutoff_datetime,
        ordinate per timestamp DESC.

        Esempio query:
        SELECT * FROM currency
        WHERE timeframe = '1h' AND base = 'ETH'
        AND CAST(`timestamp` AS DATETIME) < '2025-11-27 00:00:00'
        ORDER BY CAST(`timestamp` AS DATETIME) DESC;
        """
        # sanitize nome tabella
        safe_table = "".join(ch for ch in table_name if ch.isalnum() or ch == '_')
        if not safe_table:
            print(f"[ERROR] Nome tabella non valido: {table_name}")
            return []

        query = f"""
            SELECT *
            FROM {safe_table}
            WHERE timeframe = %s
            AND base = %s
            AND CAST(`timestamp` AS DATETIME) < %s
            ORDER BY CAST(`timestamp` AS DATETIME) DESC
        """

        try:
            # cutoff_datetime può essere una stringa 'YYYY-MM-DD HH:MM:SS'
            # oppure un oggetto datetime; mysql.connector lo gestisce
            self.cursor.execute(query, (timeframe, base, cutoff_datetime))
            rows = self.cursor.fetchall()
            columns = [col[0] for col in self.cursor.description] if self.cursor.description else []
            return [dict(zip(columns, row)) for row in rows]
        except mysql.connector.Error as err:
            print(f"Errore get_candles_before_date su {safe_table}: {err}")
            return []

    def add_timeframe(self, date_str: str, timeframe: str) -> str:
        """
        date_str: stringa nel formato 'YYYY-MM-DD HH:MM:SS'
        timeframe: uno tra '1m','5m','15m','1h','4h','1d'
        return: nuova data come stringa 'YYYY-MM-DD HH:MM:SS'
        """
        dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")

        tf_map = {
            "1m": timedelta(minutes=1),
            "5m": timedelta(minutes=5),
            "15m": timedelta(minutes=15),
            "1h": timedelta(hours=1),
            "4h": timedelta(hours=4),
            "1d": timedelta(days=1),
        }

        if timeframe not in tf_map:
            raise ValueError(f"Timeframe non supportato: {timeframe}")

        new_dt = dt + tf_map[timeframe]
        return new_dt.strftime("%Y-%m-%d %H:%M:%S")


    def is_after(self, date1: str, date2: str) -> bool:
        fmt = "%Y-%m-%d %H:%M:%S"
        return datetime.strptime(date1, fmt) > datetime.strptime(date2, fmt)


    def get_trading_context(self, base_currency: str, history_config: dict):
        """
        Recupera:
        1. Candele storiche per i timeframe richiesti.
        2. Ordine aperto (se esiste).
        3. Forecast (se esiste).
        """
        context_data = {
            "candles": {},
            "order": None,
            "forecast": []
        }

        # 1. Recupero Candele Storiche
        # Assumiamo che history_config sia tipo {"1d": 10, "1h": 20}
        for tf, limit in history_config.items():
            query = """
                SELECT * FROM currency
                WHERE base = %s AND timeframe = %s
                ORDER BY timestamp DESC
                LIMIT %s
            """
            try:
                self.cursor.execute(query, (base_currency, tf, limit))
                # Convertiamo in dizionari
                columns = [col[0] for col in self.cursor.description]
                rows = [dict(zip(columns, row)) for row in self.cursor.fetchall()]
                context_data["candles"][tf] = rows
            except mysql.connector.Error as err:
                print(f"Error fetching candles {tf}: {err}")

        # 2. Recupero Ordine Aperto
        query_order = """
            SELECT * FROM orders
            WHERE base = %s AND status = 'OPEN'
            ORDER BY created_at DESC LIMIT 1
        """
        try:
            self.cursor.execute(query_order, (base_currency,))
            res = self.cursor.fetchone()
            if res:
                columns = [col[0] for col in self.cursor.description]
                context_data["order"] = dict(zip(columns, res))
        except mysql.connector.Error as err:
             print(f"Error fetching order: {err}")

        # 3. Recupero Forecast
        # Assumiamo tabella 'forecast' con struttura simile a currency
        query_forecast = """
            SELECT * FROM forecast
            WHERE base = %s
            ORDER BY timestamp DESC LIMIT 3
        """
        try:
            self.cursor.execute(query_forecast, (base_currency,))
            columns = [col[0] for col in self.cursor.description]
            context_data["forecast"] = [dict(zip(columns, row)) for row in self.cursor.fetchall()]
        except mysql.connector.Error:
            pass # Forecast opzionale

        return context_data
