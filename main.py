from db.KrakenManager import KrakenPortfolioManager
from db.DatabaseManager import DatabaseManager
from db.MarketDataProvider import MarketDataProvider # Assicurati del percorso import corretto
from db.TimeSfmForecaster import TimeSfmForecaster
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

    # data_now = market_prov.getCandles(TARGET_PAIR, "now")


    # 3. SALVATAGGIO NEL DATABASE
    print("\n[3] Salvataggio Database...")
    db = DatabaseManager()

    # A. Salva Wallet e Ordini
    wallet_id = db.insert_wallet(summary_data)
    if wallet_id:
        db.insert_orders(wallet_id, all_positions)

    # B. Salva Dati Currency (Market Data)
    # Nota: Funziona solo se esiste la coppia in pair_limits nel DB
    # db.insert_currency_data(merged_candles, pair_info)

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

def main3():
    market_prov = MarketDataProvider()
    db = DatabaseManager()
    all_pairs_eur = market_prov.getAllPairs(quote_filter="EUR", leverage_only=True)
    for p in all_pairs_eur:
        # currency = market_prov.getCandles(p['pair'], "15m", "1mo")
        # data_1d = market_prov.getCandles(p['pair'], "1d", "5d",)
        data_4h = market_prov.getCandles(p['pair'], "4h", "1mo",)
        data_1h = market_prov.getCandles(p['pair'], "1h", "1mo", )
        # data_15m = market_prov.getCandles(p['pair'], "15m", "1d")
        # data_5m = market_prov.getCandles(p['pair'], "5m", "1d")
        # currency = market_prov.getCandles(p['pair'], "1m", "1d")
        # db.insert_currency_data(data_1d, p,"currency")
        db.insert_currency_data(data_4h, p,"currency")
        db.insert_currency_data(data_1h, p,"currency")
        # db.insert_currency_data(data_15m, p,"currency")
        # db.insert_currency_data(data_5m, p,"currency")
        # db.insert_currency_data(currency, p,"currency")

    # 2. Salva le coppie nel DB (Operazione una tantum o di aggiornamento)

    # db.insert_all_pairs(all_pairs_eur)
    db.close_connection()
    print("\n=== FINE ===")

def main4():
    market_prov = MarketDataProvider()
    forecast = TimeSfmForecaster()
    db = DatabaseManager()
    all_pairs_eur = market_prov.getAllPairs(quote_filter="EUR", leverage_only=True)
    offset = 138
    while offset > 1:
        for p in all_pairs_eur:
            data_1d = db.get_candles_with_offset('currency', "1d", p['base'], offset)
            data_4h =db.get_candles_with_offset('currency', "4h", p['base'], offset)
            data_1h =db.get_candles_with_offset('currency', "1h", p['base'], offset)
            data_15m = db.get_candles_with_offset('currency', "15m", p['base'], offset)
            data_5m = db.get_candles_with_offset('currency', "5m", p['base'], offset)
            currency = db.get_candles_with_offset('currency', "1m", p['base'], offset)
            res = forecast.predict_candles(data_1d, "1d", 3, p)
            res1 = forecast.predict_candles(data_4h, "4h", 3, p)
            res5 = forecast.predict_candles(data_1h, "1h", 3, p)
            res2 = forecast.predict_candles(data_15m, "15m", 3, p)
            res3 = forecast.predict_candles(data_5m, "5m", 3, p)
            res4 = forecast.predict_candles(currency, "1m", 3, p)
            db.insert_currency_data(res, p,"forecast")
            db.insert_currency_data(res1, p,"forecast")
            db.insert_currency_data(res2, p,"forecast")
            db.insert_currency_data(res3, p,"forecast")
            db.insert_currency_data(res4, p,"forecast")
            db.insert_currency_data(res5, p,"forecast")

        offset= offset - 1
    db.close_connection()

if __name__ == "__main__":
    # main2()
    # main3()
    # main()
    main4()
