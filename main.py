from db.KrakenManager import KrakenPortfolioManager
from db.DatabaseManager import DatabaseManager
from db.MarketDataProvider import MarketDataProvider # Assicurati del percorso import corretto

def main():
    print("=== AVVIO BOT ===")

    # 1. GESTIONE PORTAFOGLIO (KrakenManager)
    print("\n[1] Analisi Portafoglio...")
    kraken_mgr = KrakenPortfolioManager()
    kraken_mgr.get_open_orders()
    kraken_mgr.get_open_positions()
    kraken_mgr.get_normalized_portfolio()

    summary_data = kraken_mgr.get_portfolio_summary()
    all_positions = kraken_mgr.get_all_positions_data()

    # 2. GESTIONE DATI DI MERCATO (MarketDataProvider)
    print("\n[2] Analisi Mercato...")
    market_prov = MarketDataProvider()

    # Definiamo la coppia da analizzare
    TARGET_PAIR = "XBT/EUR"

    # Recupero Info Coppia
    pair_info = market_prov.getPair(TARGET_PAIR)

    # Recupero Candele (Merge di vari timeframe)
    data_1d = market_prov.getCandles(TARGET_PAIR, "1d", "1mo")
    data_4h = market_prov.getCandles(TARGET_PAIR, "4h", "5d")
    data_now = market_prov.getCandles(TARGET_PAIR, "now")

    merged_candles = market_prov.merge_candles_data(data_1d, data_4h, data_now)

    # 3. SALVATAGGIO NEL DATABASE
    print("\n[3] Salvataggio Database...")
    db = DatabaseManager()

    # A. Salva Wallet e Ordini
    wallet_id = db.insert_wallet(summary_data)
    if wallet_id:
        db.insert_orders(wallet_id, all_positions)

    # B. Salva Dati Currency (Market Data)
    # Nota: Funziona solo se esiste la coppia in pair_limits nel DB
    db.insert_currency_data(merged_candles, pair_info)

    db.close_connection()
    print("\n=== FINE ===")


def main2():
    market_prov = MarketDataProvider()
    # 1. Recupera TUTTE le coppie EUR con leva da Kraken
    all_pairs_eur = market_prov.getAllPairs(quote_filter="EUR", leverage_only=True)

    # 2. Salva le coppie nel DB (Operazione una tantum o di aggiornamento)
    db = DatabaseManager()
    db.insert_all_pairs(all_pairs_eur)
    db.close_connection()
    print("\n=== FINE ===")

if __name__ == "__main__":
    main2()
    # main()
