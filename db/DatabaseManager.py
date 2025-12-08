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

    # ... [METODI STANDARD RIMASTI INVARIATI: _get_pair_limit_id, insert_wallet, insert_orders, insert_currency_data, etc.] ...
    # Assicurati di mantenere i metodi:
    # _get_pair_limit_id, insert_wallet, insert_orders, insert_currency_data,
    # insert_all_pairs, close_connection, select_all, get_candles_with_offset,
    # get_candles_before_date, add_timeframe, is_after, get_last_candles

    # Riporto qui per chiarezza quelli che non cambiano ma servono al contesto del file
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
            if not candles_list: return
            safe_table = "".join(ch for ch in table_name if ch.isalnum() or ch == '_')
            if not safe_table: return
            p_pair = pair_info.get('pair')
            p_kr = pair_info.get('kr_pair')
            p_base = pair_info.get('base')
            p_quote = pair_info.get('quote')
            if not p_pair: return

            query = f"""
                INSERT INTO {safe_table} (
                    pair, kr_pair, base, quote,
                    timestamp, open, high, low, close, volume,
                    bid, ask, mid, spread,
                    ema_fast, ema_slow, rsi, atr, timeframe,
                    created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                ON DUPLICATE KEY UPDATE
                    open = VALUES(open), high = VALUES(high), low = VALUES(low), close = VALUES(close),
                    volume = VALUES(volume), ema_fast = VALUES(ema_fast), ema_slow = VALUES(ema_slow),
                    rsi = VALUES(rsi), atr = VALUES(atr), created_at = NOW()
            """
            data_tuples = []
            for c in candles_list:
                row = (
                    p_pair, p_kr, p_base, p_quote,
                    c.get('timestamp'), c.get('open'), c.get('high'), c.get('low'), c.get('close'), c.get('volume'),
                    c.get('bid'), c.get('ask'), c.get('mid'), c.get('spread'),
                    c.get('ema_fast'), c.get('ema_slow'), c.get('rsi'), c.get('atr'), c.get('timeframe')
                )
                data_tuples.append(row)
            try:
                self.cursor.executemany(query, data_tuples)
                self.conn.commit()
                print(f" -> Inserite/Aggiornate {self.cursor.rowcount} righe per {p_pair}")
            except mysql.connector.Error as err:
                print(f"Errore insert_currency_data: {err}")
                self.conn.rollback()

    def insert_all_pairs(self, all_pairs_list):
        if not all_pairs_list: return
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
                kr_pair = VALUES(kr_pair), base = VALUES(base), quote = VALUES(quote),
                lot_decimals = VALUES(lot_decimals), ordermin = VALUES(ordermin), pair_decimals = VALUES(pair_decimals),
                fee_volume_currency = VALUES(fee_volume_currency), fees = VALUES(fees), fees_maker = VALUES(fees_maker),
                leverage_buy = VALUES(leverage_buy), leverage_sell = VALUES(leverage_sell),
                leverage_buy_max = VALUES(leverage_buy_max), leverage_sell_max = VALUES(leverage_sell_max),
                can_leverage_buy = VALUES(can_leverage_buy), can_leverage_sell = VALUES(can_leverage_sell)
        """
        data_tuples = []
        for p in all_pairs_list:
            limits = p.get('pair_limits', {}) or {}
            row = (
                p.get('pair'), p.get('kr_pair'), p.get('base'), p.get('quote'),
                limits.get('lot_decimals'), limits.get('ordermin'), limits.get('pair_decimals'),
                limits.get('fee_volume_currency'), json.dumps(limits.get('fees', [])), json.dumps(limits.get('fees_maker', [])),
                json.dumps(limits.get('leverage_buy', [])), json.dumps(limits.get('leverage_sell', [])),
                limits.get('leverage_buy_max'), limits.get('leverage_sell_max'),
                1 if limits.get('can_leverage_buy') else 0, 1 if limits.get('can_leverage_sell') else 0
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
        safe_table = "".join(ch for ch in table_name if ch.isalnum() or ch == '_')
        if not safe_table: return []
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
        safe_table = "".join(ch for ch in table_name if ch.isalnum() or ch == '_')
        if not safe_table: return []
        query = f"""
            SELECT * FROM {safe_table}
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
            print(f"Errore get_candles_with_offset: {err}")
            return []

    def get_candles_before_date(self, table_name: str, timeframe: str, base: str, cutoff_datetime, limit: None):
        safe_table = "".join(ch for ch in table_name if ch.isalnum() or ch == '_')
        if not safe_table: return []
        query = f"""
            SELECT * FROM {safe_table}
            WHERE timeframe = %s AND base = %s
            AND CAST(`timestamp` AS DATETIME) < %s
            ORDER BY CAST(`timestamp` AS DATETIME) DESC
        """
        try:
            if limit is not None:
                query += " LIMIT %s"
                self.cursor.execute(query, (timeframe, base, cutoff_datetime, limit))
            else:
                self.cursor.execute(query, (timeframe, base, cutoff_datetime))
            rows = self.cursor.fetchall()
            columns = [col[0] for col in self.cursor.description] if self.cursor.description else []
            return [dict(zip(columns, row)) for row in rows]
        except mysql.connector.Error as err:
            print(f"Errore get_candles_before_date: {err}")
            return []

    def add_timeframe(self, date_str: str, timeframe: str) -> str:
        dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
        tf_map = {
            "1m": timedelta(minutes=1), "5m": timedelta(minutes=5), "15m": timedelta(minutes=15),
            "1h": timedelta(hours=1), "4h": timedelta(hours=4), "1d": timedelta(days=1)
        }
        if timeframe not in tf_map: raise ValueError(f"Timeframe non supportato: {timeframe}")
        return (dt + tf_map[timeframe]).strftime("%Y-%m-%d %H:%M:%S")

    def is_after(self, date1: str, date2: str) -> bool:
        fmt = "%Y-%m-%d %H:%M:%S"
        return datetime.strptime(date1, fmt) > datetime.strptime(date2, fmt)

    def get_last_candles(self, table_name: str, timeframe: str, base: str, limit: int):
        safe_table = "".join(ch for ch in table_name if ch.isalnum() or ch == '_')
        query = f"""
            SELECT * FROM {safe_table}
            WHERE timeframe = %s AND base = %s
            ORDER BY timestamp DESC
            LIMIT %s
        """
        try:
            self.cursor.execute(query, (timeframe, base, limit))
            columns = [col[0] for col in self.cursor.description]
            rows = [dict(zip(columns, row)) for row in self.cursor.fetchall()]
            return rows[::-1]
        except mysql.connector.Error as err:
            print(f"Errore get_last_candles: {err}")
            return []

    # ==============================================================================
    # BONIFICA ORDINE E UPDATE DB
    # ==============================================================================
    def _sanitize_and_update_order(self, order, current_price):
        """
        Bonifica i campi NULL dell'ordine con dati fittizi ma realistici
        e aggiorna il record nel database.
        """
        changed = False
        try:
            current_price = float(current_price)
            order_id = order.get('id')

            # --- 1. Entry Price ---
            price_entry = order.get('price_entry')
            if price_entry is None or price_entry == 0:
                price_entry = current_price
                order['price_entry'] = price_entry
                changed = True
            else:
                price_entry = float(price_entry)

            # --- 2. Qty ---
            qty = order.get('qty')
            if qty is None or qty == 0:
                qty = 50.0 / current_price if current_price > 0 else 0
                order['qty'] = qty
                changed = True
            else:
                qty = float(qty)

            # --- 3. Subtype (Buy/Sell) ---
            subtype = order.get('subtype')
            if not subtype:
                subtype = 'buy'
                order['subtype'] = subtype
                changed = True
            is_buy = (subtype.lower() == 'buy')

            # --- 4. Take Profit ---
            tp = order.get('take_profit')
            if tp is None:
                tp = price_entry * 1.02 if is_buy else price_entry * 0.98
                order['take_profit'] = tp
                changed = True

            # --- 5. Stop Loss ---
            sl = order.get('stop_loss')
            if sl is None:
                sl = price_entry * 0.99 if is_buy else price_entry * 1.01
                order['stop_loss'] = sl
                changed = True

            # --- 6. Price Avg ---
            if order.get('price_avg') is None:
                order['price_avg'] = price_entry
                changed = True

            # --- 7. Aggiornamento Prezzo Corrente ---
            order['price'] = current_price

            # --- 8. Calcolo PnL ---
            if is_buy:
                pnl = (current_price - price_entry) * qty
            else:
                pnl = (price_entry - current_price) * qty

            order['pnl'] = pnl

            # --- ESECUZIONE UPDATE ---
            query_update = """
                UPDATE orders
                SET
                    price_entry = %s,
                    qty = %s,
                    take_profit = %s,
                    stop_loss = %s,
                    price_avg = %s,
                    price = %s,
                    pnl = %s,
                    subtype = %s
                WHERE id = %s
            """
            self.cursor.execute(query_update, (
                order['price_entry'],
                order['qty'],
                order['take_profit'],
                order['stop_loss'],
                order['price_avg'],
                order['price'],
                order['pnl'],
                order['subtype'],
                order_id
            ))
            self.conn.commit()

            if changed:
                print(f" -> Ordine ID {order_id} bonificato e aggiornato.")

        except Exception as e:
            print(f"[ERR] Errore bonifica ordine {order.get('id')}: {e}")
            self.conn.rollback()

        return order

    # ==============================================================================
    # METODO GET_TRADING_CONTEXT (AGGIORNATO PROFESSIONALE)
    # ==============================================================================
    def get_trading_context(self, base_currency: str, history_config: dict, with_orders: bool = False,test_mode: bool = False):
        """
        Recupera il contesto di trading.
        Logica per il Current Price: Dinamica e basata sul Timestamp.
        Prende il prezzo della candela più recente disponibile tra tutti i timeframe richiesti.
        """
        context_data = {
            "candles": {},
            "order": None,
            "forecast": [],
            "wallet_balance": 0.0
        }

        # 1. Recupero Candele Storiche & Identificazione Prezzo Corrente
        current_price = 0.0
        last_candle_ts = None

        for tf, limit in history_config.items():
            query = """
                SELECT * FROM currency
                WHERE base = %s AND timeframe = %s
                ORDER BY timestamp DESC
                LIMIT %s
            """
            try:
                self.cursor.execute(query, (base_currency, tf, limit))
                columns = [col[0] for col in self.cursor.description]
                rows = [dict(zip(columns, row)) for row in self.cursor.fetchall()]
                context_data["candles"][tf] = rows

                # --- LOGICA PROFESSIONALE DINAMICA ---
                # Se abbiamo ottenuto righe, controlliamo se questa candela è la più recente
                if rows:
                    latest_candle = rows[0]
                    candle_ts = latest_candle['timestamp']

                    # Se non abbiamo ancora un riferimento, o se questa candela è più recente (o uguale)
                    # dell'ultima trovata, aggiorniamo il prezzo corrente.
                    # Il confronto >= gestisce il caso in cui due TF chiudono allo stesso tempo:
                    # siccome l'input è ordinato Large -> Small, l'ultima iterazione vince (corretto).
                    if last_candle_ts is None or candle_ts >= last_candle_ts:
                        last_candle_ts = candle_ts
                        current_price = float(latest_candle['close'])

            except mysql.connector.Error as err:
                print(f"Error fetching candles {tf}: {err}")

        # 2. Recupero Ordine Aperto e Bonifica
        query_order = """
            SELECT * FROM orders
            WHERE base = %s AND status = 'OPEN'
            ORDER BY created_at DESC LIMIT 1
        """
        try:
            if with_orders:
                self.cursor.execute(query_order, (base_currency,))
                res = self.cursor.fetchone()
                if res:
                    columns = [col[0] for col in self.cursor.description]
                    raw_order = dict(zip(columns, res))

                    # Bonifica solo se abbiamo un prezzo di riferimento valido
                    if current_price > 0:
                        sanitized_order = self._sanitize_and_update_order(raw_order, current_price)
                        context_data["order"] = sanitized_order
                    else:
                        context_data["order"] = raw_order
            else:
                context_data["order"] = None

        except mysql.connector.Error as err:
             print(f"Error fetching order: {err}")

        # 3. Recupero Forecast
        query_forecast = """
            SELECT * FROM forecast
            WHERE base = %s and timeframe IN (%s, %s, %s, %s)
            ORDER BY timestamp DESC LIMIT 4
        """
        try:
            self.cursor.execute(query_forecast, (base_currency,'1h+1','1h+2','15m+1','15m+2'))
            columns = [col[0] for col in self.cursor.description]
            context_data["forecast"] = [dict(zip(columns, row)) for row in self.cursor.fetchall()]
        except mysql.connector.Error:
            pass

        # 4. Wallet
        query_wallet = """
            SELECT totale_portafoglio_disponibile
            FROM wallet
            ORDER BY created_at DESC
            LIMIT 1
        """
        try:
            self.cursor.execute(query_wallet)
            res = self.cursor.fetchone()
            if res:
                context_data["wallet_balance"] = float(res[0])
        except mysql.connector.Error as err:
            print(f"Error fetching wallet: {err}")

        return context_data

    def get_trading_context_traning(
        self,
        base_currency: str,
        history_config: dict,
        pivot_timestamp,
        history_config_forecast: dict,
        forecast_forward_tf: str,
        forecast_limit: int = 3
    ):
        """
        Recupera contesto per training:
        1. Candele storiche fino al pivot_timestamp.
        2. Ordine aperto valido fino al pivot_timestamp.
        3. Forecast nel range futuro.
        """
        context_data = {
            "candles": {},
            "order": None,
            "forecast": [],
            "wallet_balance": 0.0
        }

        pivot_ts_str = pivot_timestamp.strftime("%Y-%m-%d %H:%M:%S") if isinstance(pivot_timestamp, datetime) else str(pivot_timestamp)

        # 1. Recupero Candele Storiche
        for tf, limit in history_config.items():
            context_data["candles"][tf] = self.get_candles_before_date(table_name="currency", timeframe=tf, base=base_currency, cutoff_datetime=pivot_ts_str, limit=limit)

        # 2. Recupero Ordine Aperto
        query_order = """
            SELECT * FROM orders
            WHERE base = %s AND status = 'OPEN'
              AND created_at <= %s
            ORDER BY created_at DESC LIMIT 1
        """
        try:
            self.cursor.execute(query_order, (base_currency, pivot_ts_str))
            res = self.cursor.fetchone()
            if res:
                columns = [col[0] for col in self.cursor.description]
                context_data["order"] = dict(zip(columns, res))
        except mysql.connector.Error as err:
             print(f"Error fetching order: {err}")

        # 3. Recupero Forecast
        try:
            forecast_upper = self.add_timeframe(pivot_ts_str, forecast_forward_tf)
        except ValueError as err:
            print(f"Error computing forecast upper bound: {err}")
            forecast_upper = None

        if forecast_upper:
            for tf, forecast_limit in history_config_forecast.items():
                fc_result = self.get_candles_before_date(
                    table_name="forecast",
                    timeframe=tf,
                    base=base_currency,
                    cutoff_datetime=forecast_upper,
                    limit=forecast_limit
                )
                if fc_result and len(fc_result) > 0:
                    context_data['forecast'].append(fc_result[0])

        # 4. Recupero Wallet Balance
        query_wallet = """
            SELECT totale_portafoglio_disponibile
            FROM wallet
            ORDER BY created_at DESC
            LIMIT 1
        """
        try:
            self.cursor.execute(query_wallet)
            res = self.cursor.fetchone()
            if res:
                context_data["wallet_balance"] = float(res[0])
        except mysql.connector.Error as err:
            print(f"Error fetching wallet: {err}")

        return context_data
