import os
import mysql.connector
from dotenv import load_dotenv

class DatabaseManager:
    def __init__(self):
        # 1. Carica variabili d'ambiente
        load_dotenv()

        # 2. Configurazione Connessione
        try:
            self.conn = mysql.connector.connect(
                host=os.getenv('MYSQL_HOST'),
                user=os.getenv('MYSQL_USER', 'root'), # Default a root se non c'è nell'env
                password=os.getenv('MYSQL_PASSWORD'),
                database=os.getenv('MYSQL_DATABASE'),
                port=int(os.getenv('MYSQL_PORT', 3306))
            )
            self.cursor = self.conn.cursor()
            print("--- Database: Connessione stabilita con successo ---")

        except mysql.connector.Error as err:
            print(f"Errore di connessione al Database: {err}")
            raise err

    def insert_wallet(self, summary_data):
        """
        Inserisce il riepilogo del portafoglio nella tabella Wallet.
        Input: Dizionario restituito da get_portfolio_summary()
        Output: ID del record creato
        """
        query = """
            INSERT INTO Wallet (
                total_equity_stimata,
                pnl,
                totale_portafoglio,
                totale_portafoglio_disponibile,
                totale_portafoglio_liquido,
                created_at,
                record_date
            ) VALUES (%s, %s, %s, %s, %s, NOW(), CURDATE())
        """

        # Estrai i valori dal dizionario assicurandoti l'ordine corretto
        values = (
            summary_data.get('total_equity_stimata'),
            summary_data.get('pnl'),
            summary_data.get('totale_portafoglio'),
            summary_data.get('totale_portafoglio_disponibile'),
            summary_data.get('totale_portafoglio_liquido', 0.0) # Gestione opzionale se manca la chiave
        )

        try:
            self.cursor.execute(query, values)
            self.conn.commit()
            new_id = self.cursor.lastrowid
            print(f" -> Wallet salvato con ID: {new_id}")
            return new_id

        except mysql.connector.Error as err:
            print(f"Errore insert_wallet: {err}")
            self.conn.rollback()
            return None

    def insert_orders(self, wallet_id, positions_list):
        """
        Inserisce massivamente tutte le posizioni/ordini nella tabella Orders.
        Input: wallet_id (int), lista di dizionari da get_all_positions_data()
        Output: ID dell'ultimo record inserito
        """
        if not positions_list:
            print(" -> Nessun ordine da inserire.")
            return None

        query = """
            INSERT INTO Orders (
                wallet_id, pair, kr_pair, base, quote, qty,
                price_entry, price_avg, take_profit, stop_loss,
                price, value_eur, pnl, type, subtype,
                created_at, record_date
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), CURDATE())
        """

        data_tuples = []

        for p in positions_list:
            # Preparazione tupla per ogni riga
            # Usiamo .get() per gestire eventuali None in modo sicuro
            row = (
                wallet_id,
                p.get('pair'),
                p.get('kr_pair'),
                p.get('base'),
                p.get('quote'),
                p.get('qty'),
                p.get('price_entry'), # Può essere None
                p.get('price_avg'),   # Può essere None
                p.get('take_profit'),
                p.get('stop_loss'),
                p.get('price'),
                p.get('value_eur'),
                p.get('pnl'),
                p.get('type'),
                p.get('subtype')
            )
            data_tuples.append(row)

        try:
            # executemany è molto più veloce di un ciclo di execute singoli
            self.cursor.executemany(query, data_tuples)
            self.conn.commit()

            last_id = self.cursor.lastrowid
            count = self.cursor.rowcount
            print(f" -> Inseriti {count} ordini/posizioni collegati al Wallet {wallet_id}")

            return last_id

        except mysql.connector.Error as err:
            print(f"Errore insert_orders: {err}")
            self.conn.rollback()
            return None

    def close_connection(self):
        """Chiude cursore e connessione"""
        if self.cursor: self.cursor.close()
        if self.conn: self.conn.close()
        print("--- Connessione Database Chiusa ---")

# --- ESEMPIO DI UTILIZZO ---
if __name__ == "__main__":
    # Questo blocco serve solo per testare se la connessione funziona
    # Non eseguirà insert reali a meno che tu non passi dati finti
    try:
        db = DatabaseManager()

        # Test finto (commentato per sicurezza)
        # w_id = db.insert_wallet({...dati finti...})
        # db.insert_orders(w_id, [...lista finta...])

        db.close_connection()
    except Exception as e:
        pass
