from db.KrakenManager import KrakenPortfolioManager
from db.DatabaseManager import DatabaseManager

def main():
    # 1. Recupera dati da Kraken
    kraken = KrakenPortfolioManager()
    kraken.get_open_orders()
    kraken.get_open_positions()
    kraken.get_normalized_portfolio()

    summary_data = kraken.get_portfolio_summary()
    all_positions = kraken.get_all_positions_data()

    # 2. Salva nel Database
    db = DatabaseManager()

    # A. Inserisci Wallet (Padre)
    wallet_id = db.insert_wallet(summary_data)

    if wallet_id:
        # B. Inserisci Orders (Figli) collegati al Wallet ID
        db.insert_orders(wallet_id, all_positions)

    db.close_connection()

if __name__ == "__main__":
    main()
